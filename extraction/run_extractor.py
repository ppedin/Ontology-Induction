from pathlib import Path
from google import genai
from llm.llm_keys import GEMINI_KEY
from extraction.extractors.medical_extractor import MedicalExtractor
from datasets.datasets_utils import GraphRAGBenchMedical

if __name__ == "__main__":
    gemini_client = genai.Client(api_key=GEMINI_KEY)

    extractor = MedicalExtractor(
        client=gemini_client,
        model_name="gemini-2.5-flash-lite-preview-06-17",
        output_path=Path(
            "C:/Users/paolo/Desktop/Ontology-Induction/outputs/extraction/"
            "graphragbench_medical/graphragbench_medical_extraction.parquet"
        ),
        max_workers=6,
        batch_size=3,
        thinking_budget=512,
        verbose=False,
    )

    graphragbench_medical = GraphRAGBenchMedical(
        "C:/Users/paolo/Desktop/Ontology-Induction/datasets/"
        "graphragbench_medical/graphragbench_medical_corpus",
        chunk_size=512,
    )

    extractor.extract_from_chunks(graphragbench_medical.get_documents())