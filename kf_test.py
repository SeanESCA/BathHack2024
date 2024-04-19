from kf import *
import unittest


class KF_Tests(unittest.TestCase):
        
    def test_predict_increases_P(self):
        
        kf = ExtendedKalmanFilter(x0, P0, Q, R)
        
        for i in range(10):
            
            det_P_before = np.linalg.det(kf.P)
            kf.predict(G_array[i, :], 0.01)
            det_P_after = np.linalg.det(kf.P)
            
            self.assertGreater(det_P_after, det_P_before)
            
    def test_update_decreases_P(self):
        
        kf = ExtendedKalmanFilter(x0, P0, Q, R)
            
        det_P_before = np.linalg.det(kf.P)
        kf.update(A_array[0, :])
        det_P_after = np.linalg.det(kf.P)

        self.assertLess(det_P_after, det_P_before)
        
unittest.main(argv=[''], verbosity=2, exit=False)