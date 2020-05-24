###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2020 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
# or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
# are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.
###################################################################################################

from goseek-base:latest

RUN apt-get update && apt-get install -y build-essential
RUN apt-get update && apt-get -y install cmake
RUN rm -r -f /goseek-challenge
RUN git clone https://github.com/yasnem/goseek-challenge.git /goseek-challenge --recursive

WORKDIR /goseek-challenge

COPY baselines/agents.py baselines/agents.py

COPY baselines/config/shallow-agent.yaml agent.yaml
RUN ls

WORKDIR /goseek-challenge/Open3D

RUN mkdir build
RUN apt-get install -y xorg-dev libglu1-mesa-dev libgl1-mesa-glx
RUN apt-get install -y libglew-dev
RUN apt-get install -y libglfw3-dev
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y libpng-dev
RUN apt-get install -y libsdl2-dev
RUN apt-get install -y python-dev python-tk
RUN apt-get install -y python3-dev python3-tk
RUN apt-get install -y libglu1-mesa-dev
RUN apt-get install -y libc++-7-dev
RUN apt-get install -y libc++abi-7-dev
RUN apt-get install -y ninja-build
RUN apt-get install -y libxi-dev

RUN ls

WORKDIR /goseek-challenge/Open3D/build
RUN cmake ..
RUN make -j4
RUN make install-pip-package

WORKDIR /goseek-challenge