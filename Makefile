PYTHON ?= python3

.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  probe-prepare        Create artifact tree + manifest"
	@echo "  probe-plan-shared    Materialize a shared-probe run plan"
	@echo "  probe-plan-direct    Materialize a direct-baseline run plan"
	@echo "  probe-plan-eval      Materialize an evaluation plan"
	@echo "  probe-sweep-layers   Export layer sweep plan from config"
	@echo "  probe-sweep-steps    Export video timestep sweep plan from config"
	@echo "  probe-vggt-sanity    Run a training-free VGGT -> NOVA3R decoder sanity pass"
	@echo "  probe-viz            Render preview + turntable for a probe point-cloud run"
	@echo "  probe-env            Bootstrap local conda + create/update env 'nova3r'"
	@echo "  probe-env-verify     Verify the nova3r env imports and CUDA availability"

probe-prepare:
	$(PYTHON) scripts/probe/prepare_probe_workspace.py --config configs/probe/defaults.yaml

probe-plan-shared:
	$(PYTHON) scripts/probe/train_shared_probe.py --config configs/probe/defaults.yaml

probe-plan-direct:
	$(PYTHON) scripts/probe/train_direct_baseline.py --config configs/probe/defaults.yaml

probe-plan-eval:
	$(PYTHON) scripts/probe/evaluate_probe.py --config configs/probe/defaults.yaml

probe-sweep-layers:
	$(PYTHON) scripts/probe/sweep_layers.py --config configs/probe/defaults.yaml

probe-sweep-steps:
	$(PYTHON) scripts/probe/sweep_video_timesteps.py --config configs/probe/defaults.yaml

probe-vggt-sanity:
	$(PYTHON) scripts/probe/run_vggt_to_nova3r_decoder.py

probe-viz:
	$(PYTHON) scripts/probe/visualize_run.py $(ARGS)

probe-env:
	bash scripts/probe/setup_env.sh

probe-env-verify:
	bash -lc 'if [ -x /data1/jcd_data/miniconda3/bin/conda ]; then source /data1/jcd_data/miniconda3/etc/profile.d/conda.sh; elif command -v conda >/dev/null 2>&1; then eval "$$($(command -v conda) shell.bash hook)"; else echo "conda not found"; exit 1; fi; conda activate nova3r && python scripts/probe/verify_env.py'
