import os, json
from github import Github
from openai import OpenAI

github_token  = os.getenv("GITHUB_TOKEN")        # токен для GitHub API
event_path    = os.getenv("GITHUB_EVENT_PATH")   # путь к JSON с данными PR
repo          = os.getenv("GITHUB_REPOSITORY")   # owner/repo

class GitHubClient:
    def __init__(self, token, event_path, repo):
        self._gh = Github(token)
        payload = json.load(open(event_path, encoding='utf-8'))
        pr_num = payload["pull_request"]["number"]
        self._pr = self._gh.get_repo(repo).get_pull(pr_num)

    def get_changed_files(self):
        return self._pr.get_files()

    def post_review(self, body: str):
        self._pr.create_review(body=body, event="COMMENT")

class CodexReviewer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def review(self, patches):
        instr = (
            "Ты — старший инженер-ревьювер. Проведи анализ:\n"
            "- Ищем баги и уязвимости.\n"
            "- Проверяем SOLID/KISS/DRY.\n"
            "- Предлагаем оптимизации.\n"
        )
        inp = "\n\n".join(f"Файл: {f.filename}\nПатч:\n{f.patch}" for f in patches)
        resp = self.client.responses.create(
            model="gpt-4o",
            instructions=instr,
            input=inp,
        )
        return resp.output_text.strip()

def main():
    gh = GitHubClient(
        token=os.getenv("GITHUB_TOKEN"),
        event_path=os.getenv("GITHUB_EVENT_PATH"),
        repo=os.getenv("GITHUB_REPOSITORY"),
    )
    patches = gh.get_changed_files()
    reviewer = CodexReviewer(api_key=os.getenv("OPENAI_API_KEY"))
    review = reviewer.review(patches)
    gh.post_review(review)

if __name__ == "__main__":
    main()
