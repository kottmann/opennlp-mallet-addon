/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package kottmann.opennlp.addons.mallet;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import opennlp.tools.ml.AbstractEventTrainer;
import opennlp.tools.ml.model.DataIndexer;
import opennlp.tools.ml.model.MaxentModel;
import cc.mallet.classify.Classifier;
import cc.mallet.classify.MaxEntTrainer;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelAlphabet;

public class MaxentTrainer extends AbstractEventTrainer {

  public MaxentTrainer(Map<String, String> trainParams,
      Map<String, String> reportMap) {
    super(trainParams, reportMap);
  }

  @Override
  public boolean isSortAndMerge() {
    return true;
  }

  // TODO: Turn this into a util method and share between impls ...
  private Map<String, Integer> createPrepMap(DataIndexer indexer) {
    Map<String, Integer> predMap = new HashMap<String, Integer>();

    String predLabels[] = indexer.getPredLabels();
    for (int i = 0; i < predLabels.length; i++) {
      predMap.put(predLabels[i], i);
    }

    return predMap;
  }

  @Override
  public MaxentModel doTrain(DataIndexer indexer) throws IOException {

    // NOTE: One of the issues here is that the alphabet needs to be stored
    // as well. Therefore the is no reason not to directly use the alphabet to
    // encode
    // the features.
    // It doesn't look like there is any down side to not use it directly!

    int numFeatures = indexer.getPredLabels().length;

    Alphabet dataAlphabet = new Alphabet(numFeatures);
    LabelAlphabet targetAlphabet = new LabelAlphabet();

    Collection<Instance> instances = new ArrayList<>();

    String predLabels[] = indexer.getPredLabels();
    
    int outcomes[] = indexer.getOutcomeList();
    for (int contextIndex = 0; contextIndex < indexer.getContexts().length; contextIndex++) {

      int malletFeatures[] = new int[indexer.getContexts()[contextIndex].length];
      double weights[] = new double[indexer.getContexts()[contextIndex].length];

      for (int featureIndex = 0; featureIndex < malletFeatures.length; featureIndex++) {
        malletFeatures[featureIndex] = dataAlphabet.lookupIndex(
            predLabels[indexer.getContexts()[contextIndex][featureIndex]], true);
        
        weights[featureIndex] = indexer.getNumTimesEventsSeen()[contextIndex];
      }

      FeatureVector fv = new FeatureVector(dataAlphabet, malletFeatures,
          weights);
      Instance inst = new Instance(fv, targetAlphabet.lookupLabel(
          indexer.getOutcomeLabels()[outcomes[contextIndex]], true), "name",
          "data-indexer");
      instances.add(inst);
    }

    InstanceList trainingData = new InstanceList(dataAlphabet, targetAlphabet);
    Instance inst = instances.iterator().next();

    Alphabet.alphabetsMatch(trainingData, inst);
    trainingData.addAll(instances);

    System.out.println(targetAlphabet);

    cc.mallet.classify.ClassifierTrainer trainer = new MaxEntTrainer();

    Classifier classifier = trainer.train(trainingData);

    return new ClassifierModel(classifier);
  }
}
