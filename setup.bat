@echo off
dvc destroy -f
dvc init
dvc run -n "source code" ^
		-O .hash/src.hash ^
		--always-changed ^
		-f ^
		--no-exec ^
		"python src/hashdir.py src > .hash/src.hash"
dvc run -n data ^
		-O .hash/data.hash ^
		--always-changed ^
		-f ^
		--no-exec ^
		"python src/hashdir.py ./data/source > .hash/data.hash"
dvc run -n meta ^
		-d .hash/src.hash ^
		-d .hash/data.hash ^
		-d "src/main/100 Image Meta.ipynb" ^
		-p etl.use_spark ^
		-p etl.use_CLAHE ^
		-p etl.crop_top ^
		-p etl.image_size ^
		-p etl.CLAHE_clip_limit ^
		-p etl.CLAHE_tile_size ^
		-p etl.use_segmentation ^
		-p etl.train_size ^
		-p etl.split_rand ^
		-O .hash/meta.hash ^
		-O "100 Image Meta.html" ^
		-f ^
		--no-exec ^
		"python src/run_script.py \"src/main/100 Image Meta.ipynb\" 300 && python src/hashdir.py .pkls/meta > .hash/meta.hash"
dvc run -n "feature and label" ^
		-d .hash/meta.hash ^
		-d "src/main/110 Image Data.ipynb" ^
		-p etl.use_CLAHE ^
		-p etl.crop_top ^
		-p etl.image_size ^
		-p etl.CLAHE_clip_limit ^
		-p etl.CLAHE_tile_size ^
		-p etl.use_segmentation ^
		-O data/feature ^
		-O .hash/feature.hash ^
		-O "110 Image Data.html" ^
		-f ^
		--no-exec ^
		"python src/run_script.py \"src/main/110 Image Data.ipynb\" 2200 && python src/hashdir.py ./data/feature > .hash/feature.hash"
dvc run -n "visualize image transform" ^
		-d .hash/feature.hash ^
		-d "src/main/160 Plot Transform.ipynb" ^
		-p visual.transform ^
		-p etl.use_segmentation ^
		-p etl.use_CLAHE ^
		-p etl.crop_top ^
		-p etl.image_size ^
		-p etl.CLAHE_clip_limit ^
		-p etl.CLAHE_tile_size ^
		-f ^
		--no-exec ^
		"python src/run_script.py \"src/main/160 Plot Transform.ipynb\" 300"
dvc run -n model ^
		-O .hash/model.hash ^
		--always-changed ^
		-f ^
		--no-exec ^
		"python src/hashdir.py model > .hash/model.hash"
dvc run -n "train and evaluate" ^
		-d .hash/feature.hash ^
		-d .hash/model.hash ^
		-d "src/main/200 Train.py" ^
		-p etl.image_size ^
		-p model ^
		-p train ^
		-O .hash/train.hash ^
		-f ^
		--no-exec ^
		"python -u -W ignore -m \"src.main.200 Train\" >> train.log.txt && python src/hashdir.py model > .hash/train.hash"
dvc run -n "visualize model feature" ^
		-d .hash/train.hash ^
		-d "src/main/210 Plot Feature.ipynb" ^
		-p etl.image_size ^
		-p model.tool ^
		-p model.name ^
		-p model.architect ^
		-p model.torch.in_channel ^
		-p visual.feature.example_num ^
		-p visual.feature.img_size_inch ^
		-f ^
		--no-exec ^
		"python src/run_script.py \"src/main/210 Plot Feature.ipynb\" 300"
dvc dag
dvc dag --dot > DAG.dot
dot -Tpng DAG.dot -o DAG.png
EXIT /B