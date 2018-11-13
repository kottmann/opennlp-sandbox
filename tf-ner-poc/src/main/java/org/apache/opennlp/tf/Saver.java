/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.opennlp.tf;

import java.nio.file.Path;

import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;


// The feed and target name are from the program that created the graph.

public class Saver {

  public Saver(boolean addSaveOps) {
    if (addSaveOps) {
      // TODO: Implement this!
      // It should insert the restore and save ops when the graph is created with Java.

      throw new RuntimeException("Not implemented yet!");
    }
  }

  public Saver() {
    this(true);
  }

  public void save(Session sess, Path checkpointDir) {
    try (Tensor<String> checkpointPrefix =
        Tensors.create(checkpointDir.resolve("ckpt").toString())) {
      sess.runner()
          .feed("save/Const", checkpointPrefix)
          .addTarget("save/control_dependency")
          .run();
    }
  }

  public void restore(Session sess, Path checkpointDir) {
    try (Tensor<String> checkpointPrefix =
             Tensors.create(checkpointDir.resolve("ckpt").toString())) {
      sess.runner()
          .feed("save/Const", checkpointPrefix)
          .addTarget("save/restore_all")
          .run();
    }
  }
}
