.PHONY: help fixtures verify verify-full test lint pipeline clean
PY ?= python
FIX := tests/fixtures/mini

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

fixtures:  ## build synthetic test slides
	$(PY) scripts/make_fixtures.py --out $(FIX)

test:  ## unit + integration tests
	$(PY) -m pytest tests -q

lint:  ## ruff + layering contracts (V0.3, V0.4)
	-ruff check src scripts tests
	-lint-imports --config .importlinter

verify: fixtures  ## gates V0-V6 (fast)
	@echo "=== V0.2 backends ===";        $(PY) -m src.io.slide --selftest
	@echo "=== V0.6 config hash ===";     $(PY) -m src.utils.config --print-hash configs/base.yaml configs/model_unet_effb0.yaml
	@echo "=== V1.1 slide metadata ===";  $(PY) -m src.io.slide --info $(FIX)/synth_edge.tif
	@echo "=== V2.3 tissue guards ===";   $(PY) -m src.preprocess.tissue --selftest-unimodal
	@echo "=== V3.1 annotations ===";     $(PY) -m src.io.annotations --info $(FIX)/synth_tumor.xml --slide $(FIX)/synth_tumor.tif
	@echo "=== V9.1 postproc units ===";  $(PY) -m src.infer.postproc --selftest
	@echo "=== V0.3 tests ===";           $(PY) -m pytest tests -q
	@$(MAKE) --no-print-directory v05

v05:
	@echo "=== V0.5 no hard-coded pixel constants ==="
	@grep -rnE "(min_area|radius|disk|min_size)[[:space:]]*=[[:space:]]*[0-9]+" src/ \
	  | grep -vE "(_px|# px-ok)" > /tmp/px_const.txt || true
	@test ! -s /tmp/px_const.txt && echo CLEAN || (cat /tmp/px_const.txt; exit 1)

verify-full: verify  ## adds gates V7-V10 (needs torch + data)
	$(PY) scripts/03_train.py --set train.overfit_batches=1 train.max_steps=300

pipeline:  ## run stages: make pipeline STAGES=01,02
	@for s in $$(echo $(STAGES) | tr ',' ' '); do \
	  echo "--- stage $$s ---"; $(PY) scripts/$${s}_*.py --config configs/base.yaml || exit 1; done

clean:
	rm -rf .pytest_cache **/__pycache__ /tmp/px_const.txt
