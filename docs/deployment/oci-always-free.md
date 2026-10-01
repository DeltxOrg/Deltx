# Run Deltx on OCI Always Free (Ampere A1)

This is a single-VM setup for the full `deltx dataset` command: Git history,
SonarQube Community Build, SonarScanner, and CPU-based DroidDetect inference.
Run all commands in this guide on the VM unless labeled **Mac**. The VM needs
outbound internet for OS/Python packages, Docker images, the scanner archive,
and the model's first download. SonarQube stays on VM loopback; only SSH is
open to your Mac.

## Capacity and limits

As of October 2026, OCI's [Always Free A1 allowance](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)
is 1,500 OCPU-hours and 9,000 GB-hours per month, equivalent to **one A1 VM
with 2 OCPUs and 12 GB RAM running all month**. The two AMD micro VMs have too
little RAM for this stack. The free block-volume pool is 200 GB including boot
volumes; a 100 GB boot disk leaves room for other uses. Choose an Always Free
eligible shape and disk in your tenancy's **home region**. Check the estimate
in the OCI console before creating the VM, especially on a paid account.

Older OCI material describes **4 OCPUs and 24 GB** as Always Free. Oracle's
current allowance is 2 OCPUs/12 GB for an Always Free tenancy, either in one
VM or split across two. A paid tenancy can configure a larger A1 Flex VM, but
4 OCPUs/24 GB uses the monthly free compute and memory hours in about **375
hours (15 days, 15 hours)**; usage beyond those allowances can be billed.
The [200 GB free block-volume pool](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)
is shared by boot and data volumes. Oracle also lists [10 TB/month of free
outbound transfer](https://www.oracle.com/cloud/networking/virtual-cloud-network/pricing/)
and [no charge for inbound transfer](https://www.oracle.com/cloud/price-list/).
These network allowances do not increase the free CPU or RAM budget.

This is a way to run unattended, not a guaranteed speedup. There is no free GPU
in this configuration, and 2 A1 OCPUs may be slower for DroidDetect than a
recent Mac. SonarQube alone needs roughly [2 cores and at least 4 GB RAM](https://docs.sonarsource.com/sonarqube-community-build/server-installation/server-host-requirements);
run one dataset process at a time and monitor memory/disk. OCI can reclaim
Always Free VMs it considers idle; keep a copy of finished CSVs on your Mac.

The [SonarQube server image](https://hub.docker.com/_/sonarqube) publishes an
ARM64 variant. The [official scanner Docker image](https://hub.docker.com/r/sonarsource/sonar-scanner-cli/tags)
currently publishes AMD64 only, so this repo includes a small ARM64 image built
from [SonarSource's Linux AArch64 CLI archive](https://github.com/SonarSource/sonar-update-center-properties/blob/master/scannercli.properties).
It is SHA-256 checked against [SonarSource's published checksum](https://github.com/SonarSource/sonarqube-scan-action/blob/master/sonar-scanner-version).
Verify the server image actually pulls and starts on your A1 VM before a long
run. Sonar's Docker support wording is narrower than its published ARM64 image
manifest, so treat this VM deployment as a practical setup rather than a vendor
support guarantee.

## 1. Create the VM and SSH access

In the OCI console, create an Ubuntu **24.04 ARM64** instance with shape
`VM.Standard.A1.Flex`, **2 OCPUs, 12 GB RAM**, a **100 GB boot volume**, and a
public IPv4 address. Use an SSH public key you control. Put it in a public
subnet with outbound internet access. In its network security rules allow
inbound TCP **22 only from your Mac's current public IP `/32`**; do not add
inbound rules for 9000 or 19000. OCI may report "out of host capacity" for an
Always Free shape; try another availability domain or wait, rather than silently
switching to a paid shape.

On your **Mac**, connect with the matching private key (substitute your values):

```bash
ssh -i ~/.ssh/oci_deltx ubuntu@PUBLIC_IP
```

## 2. Install Docker, Python, Poetry and Sonar host settings

On the **VM**:

```bash
sudo apt update
sudo apt install -y git rsync python3.12 python3.12-venv pipx docker.io docker-compose-v2 tmux
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
```

Log out and reconnect so Docker group membership takes effect. Membership in
the `docker` group grants root-equivalent access to this VM; grant it only to
the account you use for this work.

SonarQube embeds Elasticsearch and requires larger [Linux kernel/process
limits](https://docs.sonarsource.com/sonarqube-community-build/server-installation/pre-installation/linux).
Set the persistent kernel values; the included Compose file sets container
open-file and process limits:

```bash
printf 'vm.max_map_count=524288\nfs.file-max=131072\n' | sudo tee /etc/sysctl.d/99-deltx-sonar.conf
sudo sysctl --system
sysctl vm.max_map_count fs.file-max
docker compose version
docker info --format '{{.Architecture}}'
```

The Docker architecture should be `aarch64` or `arm64`. If `docker compose`
is missing, enable Ubuntu's `universe` repository and install
`docker-compose-v2` again.

Install [Poetry via pipx](https://python-poetry.org/docs/#installing-with-pipx),
outside the project virtual environment:

```bash
pipx install poetry
pipx ensurepath
```

Reconnect once more if `poetry` is not yet on your `PATH`.

## 3. Copy the code and the repository to analyze

Keep the Deltx checkout and target repository as separate directories. For
local Mac checkouts, these **Mac** commands copy full Git histories while
omitting the local virtual environment, secrets and model cache. Replace
`PUBLIC_IP`, key path and the target-repository path. Create the remote
directories first:

```bash
ssh -i ~/.ssh/oci_deltx ubuntu@PUBLIC_IP 'mkdir -p ~/Deltx ~/repos/Pyevolve'
rsync -az -e 'ssh -i ~/.ssh/oci_deltx' --exclude '.git/' --exclude '.venv/' --exclude '.env' --exclude 'data/models/' --exclude 'results/' --exclude '.deltx-output-*/' ./ ubuntu@PUBLIC_IP:/home/ubuntu/Deltx/
rsync -az -e 'ssh -i ~/.ssh/oci_deltx' --exclude '.venv/' --exclude '.env' /Users/praneeshsurendran/Documents/Pyevolve/ ubuntu@PUBLIC_IP:/home/ubuntu/repos/Pyevolve/
```

Run the first `rsync` from the **Deltx project root** on your Mac. Alternatively,
clone each repository from its trusted Git URL on the VM. Do not use a shallow
clone of the target repository: Deltx needs its complete commit history. If
you have a local `poetry.lock`, the `rsync` copies it; Git ignores that file,
so a fresh `git clone` will resolve dependencies again.

Back on the **VM**:

```bash
cd ~/Deltx
poetry env use python3.12
poetry install
poetry run deltx --help
```

## 4. Build the native scanner and start SonarQube

```bash
cd ~/Deltx
docker build --platform linux/arm64 -f docker/sonarqube/Dockerfile.scanner-arm64 -t deltx-sonar-scanner-arm64:8.1.0.6389 .
docker image inspect --format '{{.Architecture}}' deltx-sonar-scanner-arm64:8.1.0.6389
docker run --rm deltx-sonar-scanner-arm64:8.1.0.6389 --version
```

The image architecture must be `arm64`. `SONAR_SCANNER_PULL=false` below tells
Deltx to inspect this locally built image instead of pulling a nonexistent
registry tag. Deltx still records and runs the immutable image ID for the
dataset process.

Create a VM-local `.env` rather than copying your Mac's Sonar token:

```bash
cp .env.example .env
chmod 600 .env
nano .env
```

Set these values in `.env`:

```dotenv
SONAR_HOST_URL=http://127.0.0.1:19000
SONAR_TOKEN=
SONAR_SCANNER_IMAGE=deltx-sonar-scanner-arm64:8.1.0.6389
SONAR_SCANNER_PULL=false
SONAR_STARTUP_TIMEOUT=900
```

Start the server and check its state:

```bash
poetry run deltx sonar up
docker compose -f docker/sonarqube/compose.yaml ps
```

The managed stack binds to `127.0.0.1:19000` on the VM, and Linux host
networking lets the scanner reach that same `SONAR_HOST_URL`. If startup fails,
check `docker compose -f docker/sonarqube/compose.yaml logs --tail=100 sonarqube`,
`free -h`, `df -h`, and `sysctl vm.max_map_count`.

On your **Mac**, open an SSH tunnel in a separate terminal:

```bash
ssh -i ~/.ssh/oci_deltx -N -L 19001:127.0.0.1:19000 ubuntu@PUBLIC_IP
```

Visit `http://127.0.0.1:19001` on the Mac. Complete SonarQube's first-login
setup, change the initial admin password, and create a User analysis token.
Put the token after `SONAR_TOKEN=` in the **VM's** `.env`; keep it private.
You can close the tunnel after token creation. Do not set the VM's
`SONAR_HOST_URL` to the Mac tunnel port.

## 5. Verify both components and run the dataset

The model's first load downloads its checkpoint to `data/models/droiddetect`.
Check AI detection and Sonar scoring before a long history run:

```bash
cd ~/Deltx
poetry run deltx-detect analyze --file src/deltx/common/config.py
DELTX_RUN_SONAR_INTEGRATION=1 poetry run pytest tests/scoring/test_integration.py -v
```

The integration test creates temporary Sonar projects and checks a real scan;
it uses your VM's `.env`. It may need a few minutes on A1. If a rule-specific
assertion changes with `sonarqube:latest`, inspect the server's active Python
quality profile and the test output before trusting dataset results.

Use `tmux` so an SSH disconnect does not stop the extraction:

```bash
mkdir -p ~/Deltx/results
tmux new -s deltx-dataset
```

Inside the new `tmux` session:

```bash
cd ~/Deltx
set -o pipefail
poetry run deltx dataset ~/repos/Pyevolve --device cpu --output results/pyevolve-oci.csv 2>&1 | tee results/pyevolve-oci.log
```

Detach with **Ctrl+B, then D**. Reconnect later with
`tmux attach -t deltx-dataset`. Check progress with `tail -f
~/Deltx/results/pyevolve-oci.log`, `docker stats`, `free -h`, and `df -h`.
Use `--no-filter` only if you intend to scan every reachable checkpoint.

**Mac**, after successful completion:

```bash
scp -i ~/.ssh/oci_deltx ubuntu@PUBLIC_IP:/home/ubuntu/Deltx/results/pyevolve-oci.csv .
scp -i ~/.ssh/oci_deltx ubuntu@PUBLIC_IP:/home/ubuntu/Deltx/results/pyevolve-oci.metadata.csv .
```

The full `deltx dataset` CSV exporter in this checkout does **not** resume
after a process/VM stop. `tmux` only protects against an SSH disconnection.
The staging `.deltx-output-*` directory is removed on an ordinary abort; a
partial CSV there is not a valid resume source. Keep VM/storage backups and
run one job through to completion. The separate AI-only `deltx-extract`
Parquet command does support `--resume`.

## Stop or maintain the VM

`poetry run deltx sonar down` stops the Deltx Sonar stack and retains its
Docker volumes. Stopping the OCI VM pauses computation but keeps its boot disk;
you may still consume your free storage allocation, and an interrupted full
dataset must be restarted. Export both CSVs before deleting a VM or boot
volume. A fresh Sonar server or changed Python quality profile can change
scores, so keep the matching `*.metadata.csv` and avoid updating
`sonarqube:latest` partway through an experiment.
