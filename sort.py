import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

class KalmanBoxTracker:
    def __init__(self, bbox):
        self.bbox = np.array(bbox)
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.age = 0
        self.hits = 0
        self.no_losses = 0

    def update(self, bbox):
        self.bbox = np.array(bbox)
        self.hits += 1
        self.no_losses = 0

    def predict(self):
        self.age += 1
        if self.no_losses > 0:
            self.no_losses += 1
        return self.bbox

    def get_state(self):
        return self.bbox

KalmanBoxTracker.count = 0

class Sort:
    def __init__(self, max_age=5, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        new_trackers = []
        
        if len(self.trackers) == 0:
            for det in detections:
                self.trackers.append(KalmanBoxTracker(det))
        else:
            assigned, unassigned_dets = self.associate_detections(detections)
            
            for i, trk in enumerate(self.trackers):
                if i in assigned:
                    trk.update(detections[assigned[i]])
                    new_trackers.append(trk)
                elif trk.no_losses < self.max_age:
                    new_trackers.append(trk)
                    trk.no_losses += 1

            for i in unassigned_dets:
                new_trackers.append(KalmanBoxTracker(detections[i]))

        self.trackers = new_trackers
        return [trk.get_state() for trk in self.trackers]

    def associate_detections(self, detections):
        if len(self.trackers) == 0:
            return {}, list(range(len(detections)))
        
        cost_matrix = np.zeros((len(self.trackers), len(detections)), dtype=np.float32)
        for t, trk in enumerate(self.trackers):
            for d, det in enumerate(detections):
                cost_matrix[t, d] = np.linalg.norm(trk.get_state()[:2] - det[:2])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned = {r: c for r, c in zip(row_ind, col_ind)}
        unassigned_dets = list(set(range(len(detections))) - set(col_ind))
        return assigned, unassigned_dets
