from pathlib import Path
import counterfactual_arbiter.supervisor as shared
from .registry import MIXER_VARIANTS
ROOT=Path("/data2/yb/multimodalERC/IEMOCAP/Model_rawaux_utterance_history_residual")
def build_registry():
    out=ROOT/"formal/factorized_mixer_screen"
    return [{"experiment_id":eid,"family":"p3_factorized_mixer","module":"factorized_mixer.runner","config":str(ROOT/f"formal/factorized_mixer_configs/{eid}.json"),"output_root":str(out),"cwd":str(ROOT),"manifest":str(out/f"runs/{eid}/manifest.json")} for eid in MIXER_VARIANTS]
def external_running_ids(rows):
    import subprocess
    try: text=subprocess.check_output(["pgrep","-af","factorized_mixer.runner"],text=True)
    except subprocess.CalledProcessError: return set()
    return {r["experiment_id"] for r in rows if f"/{r['experiment_id']}.json" in text}
def main():
    shared.ROOT=ROOT; shared.build_registry=build_registry; shared.external_running_ids=external_running_ids; shared.main()
if __name__ == "__main__": main()
