import numpy as np
from collections import deque
import os
import cv2
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
from typing import Optional, List
from scipy.spatial.distance import cosine

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        ###----------------------edit-----------------------###
        self.feature = None
        ###----------------------edit-----------------------###
        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, feature):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        ###----------------------edit-----------------------###
        self.feature=feature
        ###----------------------edit-----------------------###
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, feature_extractor, global_stracks:List[STrack]=[], frame_rate=30):###<----- added feature_extractor & global_stracks attribute to constructor
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]

        ###-------------------edit-----------------------------###
        self.global_stracks = global_stracks
        self.removed_stracks = [STrack(track.tlwh, track.score) for track in global_stracks]  # type: list[STrack]
        for r_track,g_track in zip(self.removed_stracks,global_stracks):
            r_track.track_id = g_track.track_id
            r_track.feature = g_track.feature
        ###-------------------edit-----------------------------###
        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        ###------------edit------------###
        self.extractor = feature_extractor
        ###------------edit------------###

    def update(self, frame, output_results, img_info, img_size): ###<----- added frame attribute to update method
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        ###------------------------------edit------------------------------_###
        #######################################################################
        detections = [detections[i] for i in u_detection if detections[i].score>0.7]
        boxes = [track.tlbr for track in detections]
        crops = [crop_box(box=box, frame=frame) for box in boxes]
        detected_features = self.extractor(crops)

        high_removed_stracks = [track for track in self.removed_stracks if (track.score>0.7 and not track.is_activated)]
        removed_features = [track.feature for track in high_removed_stracks]
        similarity_matrix = compare_features(removed_features,detected_features)
        matched, u_removed, u_detection = filter_matches(similarity_matrix=similarity_matrix)
        print("u_detection length:",len(u_detection))
        for i_rem, i_det in matched:
            det = detections[i_det]
            track = high_removed_stracks[i_rem]
            track[i_rem].re_activate(det,self.frame_id,new_id=False)
            refind_stracks.append(track)
        
        print("REFINDS:",len(refind_stracks))
        #######################################################################
        ###------------------------------edit------------------------------_###
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            feature = detected_features[inew]  ###<----- edit
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id, feature) ###<----- passed in feature as well
            activated_starcks.append(track)
            self.global_stracks.append(track)
        print("ACTIVATED:",activated_starcks)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        print("OUTPUTS:",len(output_stracks))
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

###----------------------edit-----------------------###
#######################################################
def crop_box(frame, box):
  """
  Crops a region from a frame in RGB format based on bounding box coordinates.

  Args:
      frame: A NumPy array representing the image frame (in BGR format).
      box: A tuple containing the top-left and bottom-right coordinates of the bounding box (x_min, y_min, x_max, y_max).

  Returns:
      A NumPy array representing the cropped region in RGB format, or None if the box is invalid.
  """
  (x_min, y_min, x_max, y_max) = box

  # Check for invalid box coordinates
  if x_min >= x_max or y_min >= y_max:
    return None

  # Clamp coordinates to frame dimensions
  x_min = int(max(0, x_min))
  y_min = int(max(0, y_min))
  x_max = int(min(frame.shape[1], x_max))
  y_max = int(min(frame.shape[0], y_max))

  # Convert to RGB format
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Crop the frame in RGB
  cropped_region = frame_rgb[y_min:y_max, x_min:x_max]
  return cropped_region

def compare_features(features1: List[np.array], features2: List[np.array]):
        # 1. Preprocessing
        def preprocess_feature(feature_array):
            return feature_array.flatten()  # Or other necessary preprocessing

        # Preprocess all features
        features1_flattened = [preprocess_feature(f) for f in features1]
        features2_flattened = [preprocess_feature(f) for f in features2]

        # 2. Similarity Calculation (Example using cosine similarity)
        def cosine_similarity(a, b):
            return 1-cosine(a,b)

        print("working here")
        similarity_matrix = np.zeros((len(features1_flattened), len(features2_flattened)))
        for i in range(len(features1_flattened)):
            for j in range(len(features2_flattened)):
                similarity_matrix[i, j] = cosine_similarity(features1_flattened[i].cpu(), features2_flattened[j].cpu())

        # 3. Extract Highest Matches
        return similarity_matrix

def filter_matches(similarity_matrix):
    threshold = 0.60

    # 1. Matchings with highest similarity per row
    matches = []
    for row_index in range(similarity_matrix.shape[0]):
        row_values = similarity_matrix[row_index]
        high_sim_indices = np.where(row_values > threshold)[0]  # Indices with similarity > threshold

        if high_sim_indices.size > 0:
            best_match_index = high_sim_indices[np.argmax(row_values[high_sim_indices])]  # Index of highest
            matches.append((row_index, best_match_index))

    # 2. Rows without high similarity matches (same as before)
    rows_without_matches = np.where(~np.any(similarity_matrix > threshold, axis=1))[0]

    # 3. Columns without high similarity matches (same as before)
    cols_without_matches = np.where(~np.any(similarity_matrix > threshold, axis=0))[0]

    return matches, rows_without_matches, cols_without_matches
#######################################################
###----------------------edit-----------------------###