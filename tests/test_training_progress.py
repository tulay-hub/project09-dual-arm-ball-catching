"""Resume must continue after completed updates, including legacy checkpoints."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from training_progress import completed_updates_from_checkpoint

class TrainingProgressTests(unittest.TestCase):
    def test_legacy_stopped_checkpoint_continues_at_next_index(self):
        self.assertEqual(completed_updates_from_checkpoint({'iter':177,'infos':{'transitions':2916352}},'model_stopped_177.pt'),178)
    def test_initial_snapshot_has_no_completed_update(self):
        self.assertEqual(completed_updates_from_checkpoint({'iter':0},'model_initial.pt'),0)
    def test_explicit_metadata_handles_interruption_before_first_new_update(self):
        self.assertEqual(completed_updates_from_checkpoint({'iter':178,'infos':{'completed_updates':178}},'model_stopped_178.pt'),178)

if __name__=='__main__':unittest.main()
