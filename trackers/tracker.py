from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # sv.ByteTrack() removed — tracking is now handled by Ultralytics BoT-SORT
        # which includes OSNet Re-ID and Camera Motion Compensation (CMC)

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

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_object_tracks(self, source, read_from_stub=False, stub_path=None):
        """
        source : str (video file path) OR list of numpy arrays (pre-loaded frames).
                 Passing a file path is preferred — it lets BoT-SORT stream the video
                 directly from disk so no frames are loaded into RAM.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        # Use Ultralytics' native BoT-SORT tracker.
        # botsort.yaml enables:
        #   - OSNet Re-ID: extracts appearance embeddings from each player crop so
        #     players are re-identified after occlusion with the *same* track ID.
        #   - Camera Motion Compensation (CMC): compensates for camera panning,
        #     which is the primary cause of false ID switches in football footage.
        #
        # Passing a file-path string makes Ultralytics stream frames from disk,
        # avoiding the OutOfMemoryError that occurs when all frames are pre-loaded.
        results = self.model.track(
            source=source,
            tracker="botsort.yaml",
            persist=True,       # Persist track state across frames
            conf=0.1            # Same confidence threshold as before
        )

        for frame_num, result in enumerate(results):
            cls_names = result.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            if result.boxes is not None:
                # ------------------------------------------------------------------
                # Players & Referees — use track IDs assigned by BoT-SORT
                # ------------------------------------------------------------------
                if result.boxes.id is not None:
                    boxes     = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for bbox, track_id, cls_id in zip(boxes, track_ids, class_ids):
                        # Remap goalkeeper → player (preserve existing project logic)
                        if cls_names[cls_id] == "goalkeeper":
                            cls_id = cls_names_inv["player"]

                        if cls_id == cls_names_inv['player']:
                            tracks["players"][frame_num][track_id] = {"bbox": bbox.tolist()}

                        elif cls_id == cls_names_inv['referee']:
                            tracks["referees"][frame_num][track_id] = {"bbox": bbox.tolist()}

                # ------------------------------------------------------------------
                # Ball — no persistent ID needed; just pick the highest-conf detection
                # ------------------------------------------------------------------
                detection_supervision = sv.Detections.from_ultralytics(result)
                for frame_detection in detection_supervision:
                    bbox   = frame_detection[0].tolist()
                    cls_id = frame_detection[3]

                    if cls_id == cls_names_inv['ball']:
                        tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    # ------------------------------------------------------------------
    # Single-frame draw helpers (used by the streaming pipeline in main.py)
    # ------------------------------------------------------------------

    def draw_frame(self, frame, frame_num, tracks, team_ball_control):
        """Draw all annotations for a single frame. Returns annotated frame."""
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

        # Team ball control overlay
        frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

        return frame

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    # ------------------------------------------------------------------
    # Legacy batch method kept for compatibility (not used by main.py)
    # ------------------------------------------------------------------
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = self.draw_frame(frame, frame_num, tracks, team_ball_control)
            output_video_frames.append(frame)
        return output_video_frames