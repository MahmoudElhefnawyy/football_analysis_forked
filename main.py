from utils import save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    # -------------------------------------------------------------------------
    # CONFIG — set your input video path here
    # -------------------------------------------------------------------------
    VIDEO_PATH  = 'input_videos/08fd33_4.mp4'   # <-- change this to your video
    OUTPUT_PATH = 'output_videos/output_video.avi'
    # -------------------------------------------------------------------------

    # =========================================================================
    # PHASE 1: Tracking
    # =========================================================================
    # model.track() streams the video internally through BoT-SORT — no frames
    # are loaded into RAM.  A stub is saved so subsequent runs skip this step.
    # =========================================================================
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(
        VIDEO_PATH,                        # pass path, NOT pre-loaded frames
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )
    tracker.add_position_to_tracks(tracks)

    # =========================================================================
    # PHASE 2: Camera Movement (streams video from disk, frame by frame)
    # =========================================================================
    # Read the first frame only — needed to initialise CameraMovementEstimator.
    cap_init = cv2.VideoCapture(VIDEO_PATH)
    ret, first_frame = cap_init.read()
    cap_init.release()
    if not ret:
        raise RuntimeError(f"Cannot read video: {VIDEO_PATH}")

    camera_movement_estimator = CameraMovementEstimator(first_frame)
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement_from_path(
        VIDEO_PATH,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # =========================================================================
    # PHASE 3: Downstream processing (no frames needed)
    # =========================================================================
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # =========================================================================
    # PHASE 4: Team Colour Setup (uses only the first frame)
    # =========================================================================
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(first_frame, tracks['players'][0])

    # =========================================================================
    # PHASE 5: Ball Possession (no frame pixel data needed)
    # =========================================================================
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)

    # =========================================================================
    # PHASE 6: Streaming Draw + Write
    # =========================================================================
    # Stream the video one frame at a time:
    #   - Assign team colours (needs per-frame player crops)
    #   - Draw all annotations
    #   - Write directly to output — never store more than 1 frame in RAM
    # =========================================================================
    cap = cv2.VideoCapture(VIDEO_PATH)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 24
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_num = 0
    total_frames = len(tracks['players'])

    while frame_num < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Team assignment for this frame (needs pixel data) ---
        for player_id, track in tracks['players'][frame_num].items():
            team = team_assigner.get_player_team(frame, track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team']       = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

        # --- Draw all annotations (single-frame helpers) ---
        frame = tracker.draw_frame(frame, frame_num, tracks, team_ball_control)
        frame = camera_movement_estimator.draw_camera_movement_frame(
            frame, camera_movement_per_frame[frame_num]
        )
        frame = speed_and_distance_estimator.draw_speed_and_distance_frame(
            frame, tracks, frame_num
        )

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    print(f"Done! Output saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()