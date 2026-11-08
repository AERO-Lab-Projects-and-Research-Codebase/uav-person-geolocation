import unittest

from compare_to_ground_truth import assign_by_order


class AssignByOrderTest(unittest.TestCase):
    def test_steepest_ray_goes_with_nearest_person(self):
        # detections 0..2 look 10, 40 and 20 degrees down; people 0..2 stand 60, 20 and 35 m away
        self.assertEqual(assign_by_order([10.0, 40.0, 20.0], [60.0, 20.0, 35.0]), {1: 1, 2: 2, 0: 0})

    def test_people_too_close_in_range_to_order_are_told_apart_by_bearing(self):
        # 50 and 55 m cannot be ordered; the steeper detection looks toward 200 deg, where the 55 m person stands
        self.assertEqual(assign_by_order([30.0, 20.0], [50.0, 55.0], [200.0, 180.0], [181.0, 207.0]), {0: 1, 1: 0})
        self.assertEqual(assign_by_order([30.0, 20.0], [50.0, 75.0], [200.0, 180.0], [181.0, 207.0]), {0: 0, 1: 1})

    def test_no_pairing_when_counts_differ_or_a_ray_is_unknown(self):
        self.assertEqual(assign_by_order([10.0, 40.0], [60.0, 20.0, 35.0]), {})
        self.assertEqual(assign_by_order([10.0, None], [60.0, 20.0]), {})


if __name__ == "__main__":
    unittest.main()
