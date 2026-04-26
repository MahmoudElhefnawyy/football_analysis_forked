import os
import pickle
import numpy as np
import pandas as pd
import cv2
import torch
import supervision as sv
from ultralytics import YOLO
import lap

# Torchreid imports
# Note: torchreid needs to be installed, usually via:
# pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
from torchreid.utils import FeatureExtractor

import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class ReIDTracker:
    def __init__(self, model_path, reid_model_name='osnet_x0_25', reid_model_path=''):
        self.detector = YOLO(model_path)
        
        # Initialize Re-ID feature extractor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing ReIDTracker. Torchreid using device: {device}")
        
        self.extractor = FeatureExtractor(
            model_name=reid_model_name,
            model_path=reid_model_path,
            device=device
        )
        
        # Weights for the combined cost matrix
        self.lambda_iou = 0.8  # Weight for IoU cost
        self.lambda_reid = 0.2 # Weight for appearance cost
        
        # Active tracks: { track_id: {"bbox": [], "embedding": np.array, "lost_frames": int} }
        self.active_tracks = {}
        # Lost tracks memory (to recover after short occlusion)
        self.lost_tracks = {}
        self.next_id = 0
        self.max_lost_frames = 60 # How long to remember an occluded player

    def extract_features(self, frame, bboxes):
        """Extracts 512-D embeddings for a list of bounding boxes."""
        if len(bboxes) == 0:
            return np.array([])
            
        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            # Ensure within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            crop = frame[y1:y2, x1:x2]
            # Handle edge cases where crop is empty
            if crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
                
            crops.append(crop)
            
        # Extractor returns a torch tensor, convert to numpy
        features = self.extractor(crops)
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        return features

    def compute_iou(self, bbox_a, bbox_b):
        """Computes Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
        union = area_a + area_b - intersection

        if union == 0:
            return 0.0
        return intersection / float(union)
        
    def compute_cosine_distance(self, emb_a, emb_b):
        """Computes Cosine Distance (1 - Cosine Similarity) between two embeddings."""
        dot = np.dot(emb_a, emb_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            return 1.0 # Max distance
        similarity = dot / (norm_a * norm_b)
        return 1.0 - similarity

    def update_tracks(self, frame, bboxes, class_ids, cls_names_inv):
        """
        Manually associates new detections with active/lost tracks.
        Returns: {track_id: {"bbox": bbox.tolist(), "cls_id": cls_id}}
        """
        if len(bboxes) == 0:
            # increment lost frames
            for tid in list(self.active_tracks.keys()):
                self.lost_tracks[tid] = self.active_tracks.pop(tid)
            for tid in list(self.lost_tracks.keys()):
                self.lost_tracks[tid]["lost_frames"] += 1
                if self.lost_tracks[tid]["lost_frames"] > self.max_lost_frames:
                    del self.lost_tracks[tid]
            return {}

        # 1. Extract embeddings for current detections
        embeddings = self.extract_features(frame, bboxes)
        
        # Collect candidates to match against (active + recently lost)
        candidate_ids = list(self.active_tracks.keys()) + list(self.lost_tracks.keys())
        candidate_data = [self.active_tracks[tid] if tid in self.active_tracks else self.lost_tracks[tid] for tid in candidate_ids]
        
        num_dets = len(bboxes)
        num_cands = len(candidate_ids)
        
        # 2. Build Cost Matrix
        cost_matrix = np.zeros((num_dets, num_cands))
        
        for d in range(num_dets):
            for c in range(num_cands):
                # Motion cost (1 - IoU)
                iou = self.compute_iou(bboxes[d], candidate_data[c]["bbox"])
                motion_cost = 1.0 - iou
                
                # Appearance Cost
                app_cost = self.compute_cosine_distance(embeddings[d], candidate_data[c]["embedding"])
                
                # If IoU is exactly 0 (boxes don't touch), we rely more on appearance,
                # but we usually penalize large spatial jumps to prevent teleporting.
                # Thresholding: if they are impossibly far apart, make cost infinite
                if motion_cost > 0.95:
                    cost_matrix[d, c] = 1e6
                else:
                    cost_matrix[d, c] = self.lambda_iou * motion_cost + self.lambda_reid * app_cost

        # 3. Hungarian Optimization (LAP)
        matched_dets = set()
        matched_cands = set()
        
        if num_dets > 0 and num_cands > 0:
            # lap.lapjv minimizes the cost matrix
            cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=1.0)
            
            for d, c in enumerate(x):
                if c >= 0 and cost_matrix[d, c] < 1.0:
                    matched_dets.add(d)
                    matched_cands.add(c)
                    track_id = candidate_ids[c]
                    
                    # Move from lost to active if necessary
                    if track_id in self.lost_tracks:
                        self.active_tracks[track_id] = self.lost_tracks.pop(track_id)
                        
                    # Update active track
                    self.active_tracks[track_id]["bbox"] = bboxes[d]
                    self.active_tracks[track_id]["lost_frames"] = 0
                    self.active_tracks[track_id]["cls_id"] = class_ids[d]
                    
                    # Smooth embedding update (EMA)
                    alpha = 0.9
                    old_emb = self.active_tracks[track_id]["embedding"]
                    new_emb = embeddings[d]
                    self.active_tracks[track_id]["embedding"] = alpha * old_emb + (1 - alpha) * new_emb

        # 4. Handle unmatched detections (New Tracks)
        for d in range(num_dets):
            if d not in matched_dets:
                self.next_id += 1
                self.active_tracks[self.next_id] = {
                    "bbox": bboxes[d],
                    "embedding": embeddings[d],
                    "lost_frames": 0,
                    "cls_id": class_ids[d]
                }
                
        # 5. Handle unmatched candidates (Lost Tracks)
        for c in range(num_cands):
            if c not in matched_cands:
                track_id = candidate_ids[c]
                if track_id in self.active_tracks:
                    self.lost_tracks[track_id] = self.active_tracks.pop(track_id)
                if track_id in self.lost_tracks:
                    self.lost_tracks[track_id]["lost_frames"] += 1
                    
        # Clean up very old lost tracks
        for tid in list(self.lost_tracks.keys()):
            if self.lost_tracks[tid]["lost_frames"] > self.max_lost_frames:
                del self.lost_tracks[tid]
                
        # 6. Format output mapping
        result_dict = {}
        for tid, data in self.active_tracks.items():
            result_dict[tid] = {
                "bbox": data["bbox"].tolist(),
                "cls_id": data["cls_id"]
            }
            
        return result_dict


    # -------------------------------------------------------------------------
    # Integration Methods (Matching original Tracker interface)
    # -------------------------------------------------------------------------
    
    def get_object_tracks(self, source, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        cap = cv2.VideoCapture(source)
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            # Predict boxes natively
            results = self.detector.predict(frame, conf=0.1, verbose=False)
            result = results[0]
            
            cls_names = result.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                conf = result.boxes.conf.cpu().numpy()
                
                # Split detections: Persons (players/referees/goalkeeper) vs Ball
                person_indices = []
                ball_bbox = None
                ball_conf = -1
                
                for i, (b, cid, cnf) in enumerate(zip(boxes, class_ids, conf)):
                    name = cls_names[cid]
                    if name in ["player", "referee", "goalkeeper"]:
                        person_indices.append(i)
                    elif name == "ball":
                        if cnf > ball_conf:
                            ball_conf = cnf
                            ball_bbox = b
                            
                # Pass person detections to Re-ID Tracker
                if len(person_indices) > 0:
                    person_boxes = [boxes[i] for i in person_indices]
                    person_classes = []
                    for i in person_indices:
                        cid = class_ids[i]
                        # Remap goalkeeper to player
                        if cls_names[cid] == "goalkeeper":
                            cid = cls_names_inv["player"]
                        person_classes.append(cid)
                        
                    # Update tracks via our custom Re-ID loop
                    active_id_dict = self.update_tracks(frame, person_boxes, person_classes, cls_names_inv)
                    
                    for track_id, data in active_id_dict.items():
                        c_id = data["cls_id"]
                        if c_id == cls_names_inv["player"]:
                            tracks["players"][frame_num][track_id] = {"bbox": data["bbox"]}
                        elif c_id == cls_names_inv["referee"]:
                            tracks["referees"][frame_num][track_id] = {"bbox": data["bbox"]}
                else:
                    # Increment lost state if no detections
                    self.update_tracks(frame, [], [], cls_names_inv)
                            
                # Process ball (no ID tracking needed)
                if ball_bbox is not None:
                    tracks["ball"][frame_num][1] = {"bbox": ball_bbox.tolist()}
                    
            frame_num += 1

        cap.release()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    # ------------------------------------------------------------------
    # Data Processing & Draw Methods (Moved from original Tracker)
    # ------------------------------------------------------------------

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def draw_frame(self, frame, frame_num, tracks, team_ball_control):
        frame = frame.copy()
        player_dict  = tracks["players"][frame_num]
        ball_dict    = tracks["ball"][frame_num]
        referee_dict = tracks["referees"][frame_num]

        # Players
        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
            if player.get('has_ball', False):
                frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

        # Referees
        for _, referee in referee_dict.items():
            frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

        # Ball
        for _, ball in ball_dict.items():
            frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

        frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
        return frame

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center=(x_center, y2), axes=(int(width), int(0.35 * width)),
                    angle=0.0, startAngle=-45, endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        total = max(1, team_1_num + team_2_num)
        
        cv2.putText(frame, f"Team 1 Ball Control: {team_1_num/total * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_num/total * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame
