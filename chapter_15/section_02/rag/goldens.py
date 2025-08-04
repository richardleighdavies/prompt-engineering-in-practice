
import pickle
import json
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from dotenv import load_dotenv
load_dotenv()

with open("data/menu.json", "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

with open("data/menu.txt", "w", encoding="utf-8") as txt_file:
    txt_file.write(json.dumps(data, indent=2))
 
synthesizer = Synthesizer()

context_config = ContextConstructionConfig(
    max_contexts_per_document=1,    
    chunk_size=512,                 
    chunk_overlap=0                
)

try:
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=["data/menu.txt"],
        max_goldens_per_context=1,                    
        context_construction_config=context_config,
        include_expected_output=True                  
    )   
except Exception as e:
    print(f"Erro ao gerar goldens: {e}")
    goldens = []

with open("goldens.pkl", "wb") as f:
    pickle.dump(goldens, f)