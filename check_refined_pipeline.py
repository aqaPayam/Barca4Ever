"""Full-length generated-data CLI and official format check for refined inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import learned_possession
from games import source_frames
import joblib

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def producer_hashes(artifact):
    names=['refined_infer.py','hz25_data.py','contact_input.py','paper2023_data.py',
           'paper2023_model.py','possession_v2.py','learned_possession.py','infer_learned.py',
           'temporal_decoders.py','temporal_state.py','boundary_refinement.py',
           'torch_runtime.py','games.py','barca4ever.py','external_idsse.py']
    if artifact.get('pair_feature_mode')=='physics':names+=['boundary_physics.py','touch_physics.py']
    return {name:digest(ROOT/name) for name in names}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='models/refined_candidate_combined.joblib')
    parser.add_argument('--out',default='external_reports/refined_pipeline_check.json')
    parser.add_argument('--bundle',default='local_outputs/refined_synthetic_bundle')
    args=parser.parse_args()
    target=ROOT/args.out
    if target.exists():raise RuntimeError('Preserve previous refined pipeline-check evidence')
    bundle=(ROOT/args.bundle).resolve();local=(ROOT/'local_outputs').resolve()
    if not bundle.is_relative_to(local) or bundle==local:
        raise ValueError('Generated fixture must be a separate folder under local_outputs')
    if bundle.exists():raise RuntimeError('Preserve previous generated fixture; use a new --bundle directory')
    model_path=ROOT/args.model;model_hash=digest(model_path);artifact=joblib.load(model_path)
    before=producer_hashes(artifact)
    folder=bundle/'tactical';folder.mkdir(parents=True)
    meta=folder/'BAR-ATH_tactical_atd.xml';raw=folder/'BAR-ATH_tactical_atd.txt'
    meta.write_text('<root><Team id="h"/><Team id="a"/>'+ '<Player teamId="h"/>'*3+'<Player teamId="a"/>'*3+'</root>',encoding='utf-8')
    n=source_frames()
    if n!=199319:raise RuntimeError('Unexpected generated fixture length contract')
    with raw.open('w',encoding='utf-8',newline='\n') as stream:
        for i in range(n):
            if 90000<=i<91000:
                players=';'.join(['NaN,NaN,NaN']*6);ball='NaN,NaN,NaN'
            else:
                players='0.25,0.5,0;0.3,0.5,0;0.3,0.4,0;0.75,0.5,0;0.7,0.5,0;0.7,0.4,0'
                ball='0.25,0.5,0' if (i//500)%2==0 else '0.75,0.5,0'
            stream.write(f'{i+1}:{players}:{ball}\n')
    output=bundle/'SYNTHETIC_REFINED_CHECK.csv'
    inference_log=bundle/'inference.log';validation_log=bundle/'validation.log'
    env=os.environ.copy();env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4'
    inference_command=[sys.executable,str(ROOT/'refined_infer.py'),'--data-dir',str(bundle),
                       '--model',str(model_path),'--out',str(output),'--device','cpu']
    with inference_log.open('w',encoding='utf-8') as log:
        inference=subprocess.run(inference_command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    validation_code=None;manifest=None;manifest_ok=False
    if inference.returncode==0:
        with validation_log.open('w',encoding='utf-8') as log:
            validator=subprocess.run([sys.executable,str(ROOT/'evaluation/validate_submission.py'),str(output),'--rows',str(n)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        validation_code=validator.returncode
        manifest_path=Path(str(output)+'.manifest.json')
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            manifest_ok=(manifest['rows']==n and manifest['device']=='cpu' and manifest['model_kind']=='refined_contact'
                and manifest['model_sha256']==model_hash and manifest['output_sha256']==digest(output)
                and manifest['inference_source_sha256']==before['refined_infer.py'])
    after=producer_hashes(artifact)
    unchanged=before==after and digest(model_path)==model_hash
    passed=inference.returncode==0 and validation_code==0 and manifest_ok and unchanged
    report=dict(status='passed' if passed else 'failed',scope='Generated six-player199319-row synthetic ATD only; no competition inputs',
        device='cpu',torch_threads=4,rows=n,model_sha256=model_hash,model_sha256_after=digest(model_path),
        producer_sha256=before,producer_sha256_after=after,model_and_producer_unchanged=unchanged,
        checker_sha256=digest(__file__),validator_sha256=digest(ROOT/'evaluation/validate_submission.py'),
        generated_input_sha256={'metadata':digest(meta),'tracking':digest(raw)},
        output_sha256=digest(output) if output.exists() else None,manifest=manifest,manifest_verified=manifest_ok,
        inference_exit_code=inference.returncode,validator_exit_code=validation_code,
        inference_log=str(inference_log.relative_to(ROOT)),validation_log=str(validation_log.relative_to(ROOT)))
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n')
    if not passed:raise RuntimeError('Refined synthetic pipeline failed; inspect its local JSON report and logs')
    print('Refined full-length synthetic inference and official format check passed; hashes recorded.')
