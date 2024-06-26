
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Demonstrates loading and processing some string valued data into a
dataset that is purely numerical. Uses a portion of the UCI ML
repository census income dataset (aka Adult).
"""
import sys
import os.path

import nimble
from nimble import match

if __name__ == "__main__":
    # string manipulation to get and make paths
    projectRoot = os.path.dirname(nimble.nimblePath)
    projectData = "https://storage.googleapis.com/nimble/datasets/"

    # if a directory is given, we will output the numerical data to that location.
    if len(sys.argv) > 1:
        outDir = sys.argv[1]
    else:
        outDir = os.path.join(projectRoot, "datasets")

    pathOrig = os.path.join(projectData, "adult_income_classification_tiny.csv")
    pathOut = os.path.join(outDir, "adult_income_classification_tiny_numerical.csv")

    # we specify that we want a List object returned, and that we want the first row to
    # taken as the featureNames. Given the .csv extension on the path it will infer the
    # format of the file to be loaded, though that could be explicitly specified if desired.
    processed = nimble.data(pathOrig, featureNames=True)

    # this feature is a precalculated similarity rating. Let's not make it too easy...
    processed.features.extract('fnlwgt')

    # convert assorted features from strings to binary category columns
    processed.replaceFeatureWithBinaryFeatures('sex')
    processed.replaceFeatureWithBinaryFeatures('marital-status')
    processed.replaceFeatureWithBinaryFeatures('occupation')
    processed.replaceFeatureWithBinaryFeatures('relationship')
    processed.replaceFeatureWithBinaryFeatures('race')
    processed.replaceFeatureWithBinaryFeatures('native-country')

    # convert 'income' column (the classification label) to a numerical labels as opposed to
    # string named classes.
    processed.transformFeatureToIntegers('income')

    # scrub the rest of the string valued data -- the ones we converted are the non-redundant ones
    processed.features.delete(match.anyNonNumeric)

    # output the cleaned data set for later usage
    processed.save(pathOut, includeNames=True)
