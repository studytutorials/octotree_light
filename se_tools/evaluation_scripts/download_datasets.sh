#!/bin/bash
set -euo pipefail

# Define a dictionary of dataset names and urls.
declare -A datasets_iclnuim
declare -A datasets_tumrgbd

# User settings ################################################################
# Comment out lines for datasets you do not want to download.
# ICL-NUIM
datasets_iclnuim['living_room_traj0']='http://www.doc.ic.ac.uk/~ahanda/living_room_traj0_frei_png.tar.gz'
datasets_iclnuim['living_room_traj1']='http://www.doc.ic.ac.uk/~ahanda/living_room_traj1_frei_png.tar.gz'
datasets_iclnuim['living_room_traj2']='http://www.doc.ic.ac.uk/~ahanda/living_room_traj2_frei_png.tar.gz'
datasets_iclnuim['living_room_traj3']='http://www.doc.ic.ac.uk/~ahanda/living_room_traj3_frei_png.tar.gz'
# TUM RGBD
################################################################################
datasets_tumrgbd['fr1_xyz']='https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz'
datasets_tumrgbd['fr1_desk']='https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz'
datasets_tumrgbd['fr2_desk']='https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz'
datasets_tumrgbd['fr3_long_office_household']='https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz'



print-help() {
	echo "$0 [DIRECTORY]"
}



generate-filename-list() {
	input_dir="$1"
	output_file="$2"
	input_dir_last=$(basename "$input_dir")

	echo "# timestamp filename" > "$output_file"

	for filename in "$input_dir"/*.png ; do
		filename=$(basename "$filename")
		filename_no_ext=$(basename "$filename" .png)
		echo "$filename_no_ext $input_dir_last/$filename"
	done | sort -n >> "$output_file"
}



# Parse input arguments
case "$#" in
	0)
		output_directory="$HOME/Documents/Datasets"
		;;
	1)
		output_directory="$1"
		;;
	*)
		print-help
		exit 0
		;;
esac

# Create output directory
echo "Creating output directory $output_directory"
mkdir -p "$output_directory"



# ICL-NUIM #####################################################################
# Iterate over all ICL-NUIM datasets
for dataset in "${!datasets_iclnuim[@]}" ; do
	dataset_url=${datasets_iclnuim[$dataset]}
	dataset_filename=$(basename $dataset_url)
	dataset_directory="$output_directory/$(basename $dataset_url .tar.gz)"

	# Skip dataset if its output directory already exists
	if [ -d "$dataset_directory" ] ; then
		echo "Output directory $dataset_directory already exists, skipping $dataset."
		continue
	fi

	# Download if file does not already exist
	if [ -f "$output_directory/$dataset_filename" ] ; then
		echo "File $output_directory/$dataset_filename already exists, skipping download"
	else
		echo "Downloading $dataset from $dataset_url to $output_directory/$dataset_filename"
		wget -O "$output_directory/$dataset_filename" "$dataset_url"
	fi

	# Extract file
	echo "Extracting $output_directory/$dataset_filename into $dataset_directory"
	mkdir -p "$dataset_directory"
	tar -xzf "$output_directory/$dataset_filename" -C "$dataset_directory"
	rm "$output_directory/$dataset_filename"

	# Remove frame 0 because it has no corresponding ground truth pose
	rm "$dataset_directory/rgb/0.png"
	rm "$dataset_directory/depth/0.png"
	sed -i '/^0 depth\/0\.png 0 rgb\/0\.png$/d' "$dataset_directory/associations.txt"

	# Generate rgb.txt and depth.txt
	generate-filename-list "$dataset_directory/rgb" "$dataset_directory/rgb.txt"
	generate-filename-list "$dataset_directory/depth" "$dataset_directory/depth.txt"

	# Copy groundtruth file to groundtruth.txt
	cp "$dataset_directory"/livingRoom*.gt.freiburg "$dataset_directory/groundtruth.txt"

	echo
done



# TUM RGBD #####################################################################
# Iterate over all TUM RGBD datasets
# Iterate over all datasets
for dataset in "${!datasets_tumrgbd[@]}" ; do
	dataset_url=${datasets_tumrgbd[$dataset]}
	dataset_filename=$(basename $dataset_url)
	dataset_directory="$output_directory/$(basename $dataset_url .tgz)"

	# Skip dataset if its output directory already exists
	if [ -d "$dataset_directory" ] ; then
		echo "Output directory $dataset_directory already exists, skipping $dataset"
		continue
	fi

	# Download if file does not already exist
	if [ -f "$output_directory/$dataset_filename" ] ; then
		echo "File $output_directory/$dataset_filename already exists, skipping download"
	else
		echo "Downloading $dataset from $dataset_url to $output_directory/$dataset_filename"
		wget -O "$output_directory/$dataset_filename" "$dataset_url"
	fi

	# Extract file
	echo "Extracting $output_directory/$dataset_filename into $dataset_directory"
	tar -xzf "$output_directory/$dataset_filename" -C "$output_directory"
	rm "$output_directory/$dataset_filename"
done

