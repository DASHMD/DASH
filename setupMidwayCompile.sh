if [ "$1" == "1" ]
then
	cp CMakeLists_midway.txt CMakeLists.txt
fi
if [ "$1" == "2" ]
then
	cp CMakeLists_midway2.txt CMakeLists.txt
fi
if [ "$1" == "" ]
then
	echo "Must enter 1 or 2 for midway 1 or 2"
fi
rm -r build
mkdir build

