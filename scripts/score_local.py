import os
import argparse
from pathlib import Path

from deltx.scoring.models import IsoDimension
from deltx.scoring.pipeline import score_commit
from deltx.scoring.scoring import Normalizer
from deltx.scoring.sonar_client import SonarClient


def main() -> None:
    # Load .env file manually if it exists
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, val = line.strip().split("=", 1)
                    os.environ.setdefault(key, val)

    parser = argparse.ArgumentParser(description="Score a commit against a local SonarQube instance")
    parser.add_argument("--project-key", required=True, help="SonarQube project key")
    parser.add_argument("--token", default=os.getenv("SONAR_TOKEN"), help="SonarQube user token (defaults to SONAR_TOKEN env var)")
    parser.add_argument("--repo-path", type=Path, default=Path(os.getenv("SCAN_DIR", ".")), help="Path to the git repository (defaults to SCAN_DIR env var or .)")
    parser.add_argument("--commit", default="HEAD", help="Commit SHA or ref to score")
    args = parser.parse_args()

    if not args.token:
        print("Error: SonarQube token is required. Pass --token or set SONAR_TOKEN in .env")
        return

    # 1. Create a synthetic normalizer (in production, load a fitted one from JSON)
    normalizer = Normalizer()
    dist = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    normalizer.fit({
        IsoDimension.MAINTAINABILITY: dist,
        IsoDimension.CORRECTNESS: dist,
        IsoDimension.SECURITY: dist,
        IsoDimension.EFFICIENCY: dist,
    })

    # 2. Initialize the SonarQube client
    print(f"Connecting to SonarQube at http://localhost:9000...")
    client = SonarClient(
        base_url="http://localhost:9000",
        token=args.token
    )

    # 3. Compute the scores
    print(f"Fetching issues and calculating scores for {args.project_key} at commit {args.commit}...")
    try:
        vector = score_commit(
            component_key=args.project_key,
            source_dir=args.repo_path,
            repo_path=args.repo_path,
            commit=args.commit,
            sonar_client=client,
            normalizer=normalizer
        )
    except Exception as e:
        print(f"Error scoring commit: {e}")
        return

    # 4. Display the results
    print("\n--- Aggregated ISO/IEC 25010 Scores ---")
    print(f"Maintainability : {vector.score_maintainability:.2f} / 100")
    print(f"Correctness     : {vector.score_correctness:.2f} / 100")
    print(f"Security        : {vector.score_security:.2f} / 100")
    print(f"Efficiency      : {vector.score_efficiency:.2f} / 100")


if __name__ == "__main__":
    main()
