.PHONY: pack_model
pack_model:
	find conversion/converted -name '*py' \
		| tar -cvf nicolov_segmentation_model.tar.gz --files-from -
