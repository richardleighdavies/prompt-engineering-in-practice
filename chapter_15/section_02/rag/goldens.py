
import pickle
import json
from deepeval.synthesizer import Synthesizer
from dotenv import load_dotenv
load_dotenv()

synthesizer = Synthesizer()
with open("data/menu.json", "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

with open("data/menu.txt", "w", encoding="utf-8") as txt_file:
    txt_file.write(json.dumps(data, indent=2))

goldens = []
for i in range(3):  
    try:
        golden = synthesizer.generate_goldens_from_docs(
            document_paths=["data/menu.txt"]
        )
        goldens.append(golden)
        time.sleep(2)  
    except Exception as e:
        print(f"Error {i}: {e}")

with open("goldens.pkl", "wb") as f:
    pickle.dump(goldens, f)