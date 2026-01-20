from http import client
import os
import warnings

from dotenv import load_dotenv
from openai import OpenAI

from benchmark.index import Benchmark
from new_benchMark.Dataloader import BenchmarkDataLoader
from new_benchMark.run_test import BenchmarkTestRunner
load_dotenv()
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
if __name__ == "__main__":
    open_ai_api=os.environ.get("OPENAI_API_KEY")
    open_ai_url=os.environ.get("OPENAI_API_BASE_URL")
    model_name=os.environ.get("OPENAI_MODEL")
    client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)
    dataloader=BenchmarkTestRunner(client=client,model_name=model_name)
    dataloader.runKgConstruction()
    dataloader.hypothesisGeneration()
