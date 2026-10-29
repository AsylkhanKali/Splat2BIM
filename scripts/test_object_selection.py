import unittest

from select_object_alignment import choose


def result(valid=True,median=.25,color=.2,count=3):
    return {'status':'candidate_for_review' if valid else 'rejected_object_correspondences',
        'final_spatial_support':[{}]*count,'training_final':{'median':median},'training_color_difference':color}


class ObjectSelectionTests(unittest.TestCase):
    def test_valid_primary_is_preserved_even_if_recovery_has_lower_distance(self):
        primary=result(median=.3);recovery=result(median=.1)
        method,chosen=choose(primary,recovery)
        self.assertEqual(method,'previous_consensus');self.assertIs(chosen,primary)

    def test_recovery_is_used_only_after_primary_rejection(self):
        recovery=result();method,chosen=choose(result(False),recovery)
        self.assertEqual(method,'pair_preserving');self.assertIs(chosen,recovery)

    def test_no_invented_pose_when_both_methods_fail(self):
        self.assertEqual(choose(result(False),result(False)),(None,None))

    def test_bad_support_or_training_quality_cannot_pass_by_status_alone(self):
        for invalid in [result(count=2),result(median=.7),result(color=.5)]:
            self.assertEqual(choose(invalid,result(False)),(None,None))


if __name__=='__main__':unittest.main()
