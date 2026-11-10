"""
RunPod integration manager for LLM Tinkerer.

Provides GPU pod and serverless endpoint lifecycle management via the
RunPod GraphQL API, plus remote calls to a Flask handler running on the pod.
"""

import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"
DEFAULT_IMAGE = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
DEFAULT_PORTS = "5000/http,22/tcp"
DEFAULT_CONTAINER_DISK_GB = 50
DEFAULT_VOLUME_GB = 50

GPU_PREFERENCE = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A4000",
]


def load_runpod_api_key(project_dir: Optional[str] = None) -> str:
    """Load a RunPod API key from env or local files. Never logs the value."""
    env_key = os.getenv("RUNPOD_API_KEY", "").strip()
    if env_key:
        return env_key

    base = Path(project_dir) if project_dir else Path(__file__).resolve().parent
    candidates = [
        base / ".env",
        base / "runpod_api_key.txt",
        base / "runpod.key",
        Path.home() / ".runpod" / "api_key",
    ]
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                name, value = line.split("=", 1)
                if name.strip() in {"RUNPOD_API_KEY", "RUNPOD_KEY", "API_KEY"}:
                    return value.strip().strip('"').strip("'")
            else:
                return line
    return ""


def proxy_url_from_pod_id(pod_id: str, port: int = 5000) -> str:
    return f"https://{pod_id.strip()}-{port}.proxy.runpod.net"


@dataclass
class PodInfo:
    id: str
    name: str
    image_name: str
    gpu_type: str
    desired_status: str
    ports: str
    cost_per_hr: float
    proxy_url: Optional[str] = None
    machine_id: Optional[str] = None


class RunPodManager:
    """Manages RunPod GPU pods and serverless endpoints for remote LLM tinkering."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY", "")
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def _graphql(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a GraphQL query/mutation against the RunPod API."""
        if not self.api_key:
            raise ValueError("RunPod API key is required.")
        url = f"{RUNPOD_GRAPHQL_URL}?api_key={self.api_key}"
        payload: Dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        resp = self._session.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"RunPod GraphQL error: {data['errors']}")
        return data.get("data", {})

    def list_gpu_types(self) -> List[Dict[str, Any]]:
        """Return available GPU types with pricing and availability."""
        query = """
        query {
          gpuTypes {
            id
            displayName
            manufacturer
            memoryInGb
            cudaCores
            secureCloud
            communityCloud
            securePrice
            communityPrice
            communitySpotPrice
            secureSpotPrice
            stockStatus
          }
        }
        """
        data = self._graphql(query)
        return data.get("gpuTypes", [])

    def list_pods(self) -> List[Dict[str, Any]]:
        """Return all pods owned by the user."""
        query = """
        query {
          myself {
            pods {
              id
              name
              desiredStatus
              imageName
              ports
              costPerHr
              machineId
              runtime {
                uptimeInSeconds
                gpus {
                  id
                  gpuUtilPercent
                  memoryUtilPercent
                }
              }
            }
          }
        }
        """
        data = self._graphql(query)
        return data.get("myself", {}).get("pods", [])

    def get_pod(self, pod_id: str) -> Dict[str, Any]:
        query = """
        query($input: PodFilter) {
          pod(input: $input) {
            id
            name
            desiredStatus
            imageName
            ports
            costPerHr
            machineId
            runtime {
              uptimeInSeconds
              gpus {
                id
                gpuUtilPercent
                memoryUtilPercent
              }
              container {
                cpuPercent
                memoryPercent
              }
            }
          }
        }
        """
        variables = {"input": {"podId": pod_id}}
        data = self._graphql(query, variables)
        return data.get("pod", {})

    def _build_proxy_url(self, pod_id: str, port: int = 5000) -> str:
        return proxy_url_from_pod_id(pod_id, port)

    def _build_docker_args(
        self, handler_b64: str, requirements_b64: str, extra_files_b64: Optional[Dict[str, str]] = None
    ) -> str:
        """Build a dockerArgs startup script that installs deps and starts the handler."""
        extra_files_b64 = extra_files_b64 or {}
        decode_steps = []
        for fname, fbody in extra_files_b64.items():
            decode_steps.append(f"echo {fbody} | base64 -d > /workspace/{fname}")
        decode_steps.append(f"echo {handler_b64} | base64 -d > /workspace/runpod_handler.py")
        decode_steps.append(f"echo {requirements_b64} | base64 -d > /workspace/requirements-pod.txt")
        decode_steps.append(
            "pip install --no-cache-dir --ignore-installed -r /workspace/requirements-pod.txt > /workspace/pip.log 2>&1"
        )
        decode_steps.append("cd /workspace && python runpod_handler.py > /workspace/server.log 2>&1")
        script = f"bash -c '{' && '.join(decode_steps)}'"
        return script

    def create_pod(
        self,
        name: str = "llm-tinkerer",
        gpu_type: str = "",
        image_name: str = DEFAULT_IMAGE,
        project_dir: Optional[str] = None,
        ports: str = DEFAULT_PORTS,
        container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
        volume_gb: int = DEFAULT_VOLUME_GB,
        env: Optional[List[Dict[str, str]]] = None,
    ) -> PodInfo:
        """Create a new GPU pod, optionally embedding a handler via dockerArgs."""
        docker_args = ""
        if project_dir:
            project_path = Path(project_dir)
            handler_src = project_path / "runpod_handler.py"
            req_src = project_path / "runpod_requirements.txt"
            if not handler_src.exists():
                raise FileNotFoundError(f"Handler not found: {handler_src}")
            if not req_src.exists():
                raise FileNotFoundError(f"Requirements not found: {req_src}")
            handler_b64 = base64.b64encode(handler_src.read_bytes()).decode()
            req_b64 = base64.b64encode(req_src.read_bytes()).decode()
            extra_files = {}
            for extra_name in (
                "weight_inversion_implementation.py",
                "neuron_explorer.py",
                "weight_space.py",
                "fact_editor.py",
                "eval_probe.py",
                "vlm_fact_editor.py",
            ):
                extra_file = project_path / extra_name
                if extra_file.exists():
                    extra_files[extra_name] = base64.b64encode(
                        extra_file.read_bytes()
                    ).decode()
            docker_args = self._build_docker_args(handler_b64, req_b64, extra_files)

        selected_gpu = gpu_type or GPU_PREFERENCE[0]
        safe_name = name.replace('"', "'").replace("\\", "")
        env_list = env or []
        env_str = ", ".join(
            f'{{ key: "{e["key"]}", value: "{e["value"]}" }}' for e in env_list
        )
        env_field = f"env: [{env_str}]" if env_str else ""
        args_field = f'dockerArgs: "{docker_args}"' if docker_args else ""

        query = f"""
        mutation {{
          podFindAndDeployOnDemand(
            input: {{
              cloudType: ALL
              gpuCount: 1
              volumeInGb: {volume_gb}
              containerDiskInGb: {container_disk_gb}
              gpuTypeId: "{selected_gpu}"
              name: "{safe_name}"
              imageName: "{image_name}"
              ports: "{ports}"
              volumeMountPath: "/workspace"
              {env_field}
              {args_field}
            }}
          ) {{
            id
            imageName
            machineId
            name
            desiredStatus
            ports
            costPerHr
          }}
        }}
        """
        data = self._graphql(query)
        pod_data = data.get("podFindAndDeployOnDemand", {})
        pod_id = pod_data.get("id", "")
        proxy_url = self._build_proxy_url(pod_id) if pod_id else None
        return PodInfo(
            id=pod_id,
            name=pod_data.get("name", name),
            image_name=pod_data.get("imageName", image_name),
            gpu_type=selected_gpu,
            desired_status=pod_data.get("desiredStatus", ""),
            ports=pod_data.get("ports", ports),
            cost_per_hr=pod_data.get("costPerHr", 0.0),
            proxy_url=proxy_url,
            machine_id=pod_data.get("machineId"),
        )

    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a pod by ID."""
        query = f"""
        mutation {{
          podTerminate(input: {{ podId: "{pod_id}" }})
        }}
        """
        self._graphql(query)
        return True

    def stop_pod(self, pod_id: str) -> Dict[str, Any]:
        query = f"""
        mutation {{
          podStop(input: {{ podId: "{pod_id}" }}) {{
            id
            desiredStatus
          }}
        }}
        """
        return self._graphql(query)

    def resume_pod(self, pod_id: str) -> Dict[str, Any]:
        query = f"""
        mutation {{
          podResume(input: {{ podId: "{pod_id}" }}) {{
            id
            desiredStatus
          }}
        }}
        """
        return self._graphql(query)

    def wait_for_pod_ready(
        self,
        proxy_url: str,
        timeout: float = 300.0,
        poll_interval: float = 10.0,
    ) -> bool:
        """Poll the remote handler's /health endpoint until it responds."""
        start = time.time()
        health_url = f"{proxy_url}/health"
        while time.time() - start < timeout:
            try:
                resp = self._session.get(health_url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "healthy":
                        return True
            except Exception as e:
                logger.debug(f"Health poll waiting: {e}")
            time.sleep(poll_interval)
        return False

    def find_existing_pod(self, name: str = "llm-tinkerer") -> Optional[PodInfo]:
        """Find a running pod with the given name."""
        pods = self.list_pods()
        for pod in pods:
            if pod.get("name") == name:
                status = pod.get("desiredStatus", "").upper()
                if status in {"RUNNING", "CREATED"}:
                    pod_id = pod.get("id", "")
                    return PodInfo(
                        id=pod_id,
                        name=pod.get("name", name),
                        image_name=pod.get("imageName", ""),
                        gpu_type="",
                        desired_status=pod.get("desiredStatus", ""),
                        ports=pod.get("ports", ""),
                        cost_per_hr=pod.get("costPerHr", 0.0),
                        proxy_url=self._build_proxy_url(pod_id),
                        machine_id=pod.get("machineId"),
                    )
        return None

    def ensure_pod_ready(
        self,
        name: str = "llm-tinkerer",
        gpu_type: str = "",
        project_dir: Optional[str] = None,
    ) -> PodInfo:
        """Return an existing healthy pod, or create a new one and wait for it."""
        existing = self.find_existing_pod(name)
        if existing:
            logger.info(f"Found existing pod {existing.id}")
            if self.wait_for_pod_ready(existing.proxy_url, timeout=60, poll_interval=5):
                return existing
            logger.info(f"Existing pod {existing.id} not healthy, terminating it")
            self.terminate_pod(existing.id)
            time.sleep(5)
        pod = self.create_pod(name=name, gpu_type=gpu_type, project_dir=project_dir)
        logger.info(f"Created pod {pod.id}, waiting for handler to start")
        if self.wait_for_pod_ready(pod.proxy_url):
            return pod
        raise RuntimeError(f"Pod {pod.id} did not become ready in time")

    # --- Remote handler calls ---

    def remote_health(self, proxy_url: str) -> Dict[str, Any]:
        resp = self._session.get(f"{proxy_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def remote_load_model(self, proxy_url: str, model_name: str, use_safetensors: bool = True) -> Dict[str, Any]:
        resp = self._session.post(
            f"{proxy_url}/load_model",
            json={"model_name": model_name, "use_safetensors": use_safetensors},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    def remote_generate(
        self,
        proxy_url: str,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        resp = self._session.post(
            f"{proxy_url}/generate",
            json={"prompt": prompt, "max_length": max_length, "temperature": temperature},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    def remote_invert_weights(
        self,
        proxy_url: str,
        inversion_type: str,
        multiplier: float,
        layer_start: int,
        layer_end: int,
        custom_rules: str,
    ) -> Dict[str, Any]:
        resp = self._session.post(
            f"{proxy_url}/invert_weights",
            json={
                "inversion_type": inversion_type,
                "multiplier": multiplier,
                "layer_start": layer_start,
                "layer_end": layer_end,
                "custom_rules": custom_rules,
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    def remote_restore_weights(self, proxy_url: str) -> Dict[str, Any]:
        resp = self._session.post(f"{proxy_url}/restore_weights", timeout=300)
        resp.raise_for_status()
        return resp.json()

    def remote_list_layers(self, proxy_url: str) -> List[str]:
        resp = self._session.get(f"{proxy_url}/list_layers", timeout=30)
        resp.raise_for_status()
        return resp.json().get("layers", [])

    def remote_get_structure(self, proxy_url: str) -> Dict[str, Any]:
        resp = self._session.get(f"{proxy_url}/structure", timeout=30)
        resp.raise_for_status()
        return resp.json()

    # --- Neuron explorer calls ---

    def remote_neuron_info(self, proxy_url: str) -> Dict[str, Any]:
        resp = self._session.get(f"{proxy_url}/neuron_info", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def remote_neuron_steer_generate(
        self,
        proxy_url: str,
        prompt: str,
        layer: int,
        neuron: Optional[int] = None,
        dose: float = 0.0,
        dose_mode: str = "rel",
        max_new_tokens: int = 40,
        temperature: float = 0.7,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        resp = self._session.post(
            f"{proxy_url}/neuron_steer_generate",
            json={
                "prompt": prompt,
                "layer": layer,
                "neuron": neuron,
                "dose": dose,
                "dose_mode": dose_mode,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "seed": seed,
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    def remote_neuron_sweep_start(self, proxy_url: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._session.post(f"{proxy_url}/neuron_sweep", json=cfg, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def remote_neuron_sweep_status(self, proxy_url: str) -> Dict[str, Any]:
        resp = self._session.get(f"{proxy_url}/neuron_sweep_status", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def remote_eval(self, proxy_url: str, label: str = "") -> Dict[str, Any]:
        r = requests.post(
            f"{proxy_url}/eval", json={"label": label}, timeout=600)
        return r.json()

    def remote_fact_write(
        self, proxy_url: str, asks: List[str], answer: str,
        contrast: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        r = requests.post(
            f"{proxy_url}/fact_write",
            json={"asks": asks, "answer": answer,
                  "contrast": contrast or []},
            timeout=1800)
        return r.json()

    def remote_fact_compare(
        self, proxy_url: str, q: str = "", n: int = 60,
    ) -> Dict[str, Any]:
        r = requests.post(
            f"{proxy_url}/fact_compare", json={"q": q, "n": n},
            timeout=300)
        return r.json()

    def remote_fact_undo(self, proxy_url: str) -> Dict[str, Any]:
        return requests.post(f"{proxy_url}/fact_undo", timeout=60).json()

    def remote_fact_commit(self, proxy_url: str) -> Dict[str, Any]:
        return requests.post(f"{proxy_url}/fact_commit", timeout=60).json()

    def remote_ws_check(self, proxy_url: str, name: str) -> Dict[str, Any]:
        r = requests.post(
            f"{proxy_url}/ws_check", json={"name": name}, timeout=120)
        return r.json()

    def remote_ws_load_ref(self, proxy_url: str, name: str) -> Dict[str, Any]:
        r = requests.post(
            f"{proxy_url}/ws_load_ref", json={"name": name}, timeout=1800)
        return r.json()

    def remote_ws_merge(
        self, proxy_url: str, mode: str = "task_add", lam: float = 1.0,
        name_filter: str = "",
    ) -> Dict[str, Any]:
        r = requests.post(
            f"{proxy_url}/ws_merge",
            json={"mode": mode, "lam": lam, "filter": name_filter},
            timeout=900)
        return r.json()

    def remote_ws_undo(self, proxy_url: str) -> Dict[str, Any]:
        return requests.post(f"{proxy_url}/ws_undo", timeout=120).json()

    def remote_weight_stats(self, proxy_url: str, name: str) -> Dict[str, Any]:
        r = requests.post(
            f"{proxy_url}/weight_stats", json={"name": name}, timeout=60)
        return r.json()

    def remote_sculpt_op(
        self, proxy_url: str, kind: str, layer: int, neuron: int,
        value: float = 0.0,
    ) -> Dict[str, Any]:
        r = requests.post(
            f"{proxy_url}/sculpt_op",
            json={"kind": kind, "layer": layer, "neuron": neuron,
                  "value": value},
            timeout=180)
        return r.json()

    def remote_undo_sculpt(self, proxy_url: str, n: int = 1) -> Dict[str, Any]:
        return requests.post(
            f"{proxy_url}/undo_sculpt", json={"n": n}, timeout=60).json()

    def remote_vlm_fact_write(
        self, proxy_url: str, asks: List[Dict[str, Any]], answer: str,
        contrast: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """asks/contrast items: {"image": <b64 png> or None, "q": str}"""
        r = requests.post(
            f"{proxy_url}/vlm_fact_write",
            json={"asks": asks, "answer": answer,
                  "contrast": contrast or []},
            timeout=3600)
        return r.json()

    def remote_vlm_fact_compare(
        self, proxy_url: str, image_b64: Optional[str], q: str, n: int = 60,
    ) -> Dict[str, Any]:
        r = requests.post(
            f"{proxy_url}/vlm_fact_compare",
            json={"image": image_b64, "q": q, "n": n}, timeout=600)
        return r.json()

    def remote_vlm_fact_undo(self, proxy_url: str) -> Dict[str, Any]:
        return requests.post(
            f"{proxy_url}/vlm_fact_undo", timeout=60).json()

    def remote_save_model(self, proxy_url: str) -> Dict[str, Any]:
        return requests.post(
            f"{proxy_url}/save_model", timeout=900).json()

    def remote_download_model(
        self, proxy_url: str, dest_dir: str,
    ) -> Dict[str, Any]:
        """Stream edited_model.zip from the pod and extract to dest_dir."""
        import io
        import shutil
        import tempfile
        import zipfile

        with requests.get(
                f"{proxy_url}/download_model", stream=True,
                timeout=3600) as r:
            if r.status_code != 200:
                try:
                    return r.json()
                except Exception:
                    return {"error": f"HTTP {r.status_code}"}
            zpath = os.path.join(
                tempfile.gettempdir(), "edited_model.zip")
            with open(zpath, "wb") as f:
                shutil.copyfileobj(r.raw, f, length=1024 * 1024)
        os.makedirs(dest_dir, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest_dir)
        return {"extracted": dest_dir,
                "files": len(zipfile.ZipFile(zpath).namelist())}

    def remote_neuron_sweep_result(self, proxy_url: str) -> str:
        resp = self._session.get(f"{proxy_url}/neuron_sweep_result", timeout=600)
        resp.raise_for_status()
        return resp.text

    # --- Serverless endpoints ---

    def list_endpoints(self) -> List[Dict[str, Any]]:
        query = """
        query {
          myself {
            endpoints {
              id
              name
              gpuIds
              workersMin
              workersMax
              idleTimeout
              templateId
            }
          }
        }
        """
        data = self._graphql(query)
        return data.get("myself", {}).get("endpoints", [])

    def create_serverless_endpoint(
        self,
        name: str,
        template_id: str,
        gpu_ids: str = "AMPERE_24",
        workers_min: int = 0,
        workers_max: int = 3,
        idle_timeout: int = 5,
    ) -> Dict[str, Any]:
        """Create a serverless endpoint from a template."""
        query = f"""
        mutation {{
          saveEndpoint(
            input: {{
              gpuIds: "{gpu_ids}"
              idleTimeout: {idle_timeout}
              name: "{name}"
              templateId: "{template_id}"
              workersMin: {workers_min}
              workersMax: {workers_max}
            }}
          ) {{
            id
            name
            gpuIds
            workersMin
            workersMax
            idleTimeout
          }}
        }}
        """
        return self._graphql(query)

    def run_serverless_job(self, endpoint_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a job to a serverless endpoint via the REST API."""
        url = f"https://api.runpod.io/v2/{endpoint_id}/run"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = self._session.post(
            url,
            json={"input": payload},
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
