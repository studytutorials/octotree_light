all: release

release:
	mkdir -p build/release
	cd build/release && cmake -DCMAKE_BUILD_TYPE=Release $(CMAKE_ARGUMENTS) ../..
	$(MAKE) -C build/release $(MFLAGS) $(SPECIFIC_TARGET)

release-with-debug:
	mkdir -p build/relwithdebinfo
	cd build/relwithdebinfo && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo $(CMAKE_ARGUMENTS) ../..
	$(MAKE) -C build/relwithdebinfo $(MFLAGS) $(SPECIFIC_TARGET)

debug:
	mkdir -p build/debug/logs
	cd build/debug && cmake -DCMAKE_BUILD_TYPE=Debug $(CMAKE_ARGUMENTS) ../..
	$(MAKE) -C build/debug $(MFLAGS) $(SPECIFIC_TARGET)

stats:
	mkdir -p build/release/logs
	cd build/release && cmake -DSTATS=ON -DCMAKE_BUILD_TYPE=Release $(CMAKE_ARGUMENTS) ../..
	$(MAKE) -C build/release $(MFLAGS) $(SPECIFIC_TARGET)

install:
	$(MAKE) -C build/release $(MFLAGS) install

uninstall:
	$(MAKE) -C build/release $(MFLAGS) uninstall



#### TESTING ####
build-tests:
	$(MAKE) -C se_shared/test $(MFLAGS)
	$(MAKE) -C se_core/test $(MFLAGS)
	$(MAKE) -C se_voxel_impl/test $(MFLAGS)
	$(MAKE) -C se_denseslam/test $(MFLAGS)

test: build-tests
	$(MAKE) -C se_shared/test $(MFLAGS) test
	$(MAKE) -C se_core/test $(MFLAGS) test
	$(MAKE) -C se_voxel_impl/test $(MFLAGS) test
	$(MAKE) -C se_denseslam/test $(MFLAGS) test

clean-tests:
	$(MAKE) -C se_shared/test $(MFLAGS) clean
	$(MAKE) -C se_core/test $(MFLAGS) clean
	$(MAKE) -C se_voxel_impl/test $(MFLAGS) clean
	$(MAKE) -C se_denseslam/test $(MFLAGS) clean



#### DATA SET GENERATION ####

living_room_traj%_loop.raw : living_room_traj%_loop
	if test -x ./build/release/thirdparty/scene2raw ; then echo "..." ; else echo "do make before"; false ; fi
	./build/release/thirdparty/scene2raw living_room_traj$(*F)_loop living_room_traj$(*F)_loop.raw

living_room_traj%_loop :
	mkdir $@
	cd $@ ; wget http://www.doc.ic.ac.uk/~ahanda/$@.tgz; tar xzf $@.tgz

livingRoom%.gt.freiburg :
	echo  "Download ground truth trajectory..."
	if test -x $@ ; then echo "Done" ; else wget http://www.doc.ic.ac.uk/~ahanda/VaFRIC/$@ ; fi

live.log :
	./build/release/kfusion-qt-openmp $(live)

demo-ofusion:
	./build/release/kfusion-main-openmp --image-resolution-ratio 2 --fps 0 --block-read False --input-file /data/ev314/data/living_room_traj2_frei_png/scene.raw --icp-threshold 1e-05 --mu 0.008 --init-pose 0.34,0.5,0.24 --integration-rate 1 --volume-size 5 -B 8 --tracking-rate 1 --map-size 512 --pyramid-levels 10,5,4 --rendering-rate 1 -k 481.2,-480,320,240

demo-kfusion:
	./build/release/kfusion-main-openmp --image-resolution-ratio 2 --fps 0 --block-read False --input-file /data/ev314/data/living_room_traj2_frei_png/scene.raw --icp-threshold 1e-05 --mu 0.1 --init-pose 0.34,0.5,0.24 --integration-rate 1 --volume-size 5 -B 8 --tracking-rate 1 --map-size 512 --pyramid-levels 10,5,4 --rendering-rate 1 -k 481.2,-480,320,240


#### GENERAL GENERATION ####

doc :
	doxygen

clean :
	rm -rf build
cleanall :
	rm -rf build
	rm -rf living_room_traj*_loop livingRoom*.gt.freiburg living_room_traj*_loop.raw
	rm -f *.log
	rm -rf doc/html


.PHONY : clean bench test all validate doc install uninstall

.PRECIOUS: living_room_traj%_loop livingRoom%.gt.freiburg living_room_traj%_loop.raw

