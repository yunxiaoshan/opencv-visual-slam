#CMPT 742 Fall2018 Assignment 05
### Group menbers: Jiayi Zhou, Lei Pan

----------------------------------

## Note:
- Helper: http://answers.opencv.org/question/100382/convert-vectorpoint-to-mat-with-2-columns/
- In report: threshold = 0.1 explaination
    - running speed, cpu + put klp + slam
    - threshold, image and curve
- WTF: come from here? https://github.com/ychemli/Fundamental-Matrix/blob/master/Fundamental.cpp

## Implementation Steps
1. To make build folder/ change working directory
    - run build.sh file with the opencv and sov zips
    - change the dataset path to the current path in the default config file
2. To run KLT
    - To compile, in build, run `make`
    - To run, in build, run `./klt ../<image 1 name> ../<image 2 name>`
3. To run frontend
    - To compile, in build, run `make`
    - To run, in build, run `./vo_offline ../config/default.yml`
