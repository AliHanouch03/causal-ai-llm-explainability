"""test if llm is reachable and returns something"""

from dotenv import load_dotenv
load_dotenv()

from llm_clients import list_llms, get_client


def test_all_llms():
    test_prompt = "In one sentence, what is a causal graph?"
    
    for llm_name in list_llms():
        print(f"\n--- {llm_name} ---")
        try:
            client = get_client(llm_name)
            response = client.generate(test_prompt)
            print(response[:200])
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_all_llms()