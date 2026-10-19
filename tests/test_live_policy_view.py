import sys,unittest,json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from io import BytesIO
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from live_policy_view import LivePolicyView

class LivePolicyViewTests(unittest.TestCase):
    def test_real_http_page_status_and_jpeg_publication(self):
        with TemporaryDirectory() as tmp:
            viewer=LivePolicyView(0,'model_test.pt',tmp)
            try:
                viewer.publish(np.full((12,16,3),[180,20,10],dtype=np.uint8),{'episodes':2,'successes':1,'phase':'收手抓握'})
                with urlopen(viewer.url+'status.json',timeout=2) as response:
                    data=json.load(response)
                self.assertEqual(data['episodes'],2);self.assertEqual(data['successes'],1);self.assertEqual(data['frame'],1)
                with urlopen(viewer.url+'frame.jpg',timeout=2) as response:
                    image=Image.open(BytesIO(response.read()));self.assertEqual(image.size,(16,12))
                with urlopen(viewer.url,timeout=2) as response:self.assertIn('model_test.pt',response.read().decode())
                manifest=json.loads((Path(tmp)/'live_view.json').read_text());self.assertFalse(manifest['training_updates'])
            finally:viewer.close()

if __name__=='__main__':unittest.main()
