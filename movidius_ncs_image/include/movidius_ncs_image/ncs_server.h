/*
 * Copyright (c) 2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MOVIDIUS_NCS_IMAGE_NCS_SERVER_H
#define MOVIDIUS_NCS_IMAGE_NCS_SERVER_H

#include <string>
#include <vector>

#include <ros/ros.h>

#include <movidius_ncs_msgs/ClassifyObject.h>
#include <movidius_ncs_msgs/DetectObject.h>
#include "movidius_ncs_lib/ncs.h"

namespace movidius_ncs_image
{
class NCSServer
{
public:
  explicit NCSServer(ros::NodeHandle& nh);

private:
  void getParameters();
  void init();

  bool cbClassifyObject(movidius_ncs_msgs::ClassifyObject::Request& request,
                        movidius_ncs_msgs::ClassifyObject::Response& response);
  bool cbDetectObject(movidius_ncs_msgs::DetectObject::Request& request,
                      movidius_ncs_msgs::DetectObject::Response& response);
  ros::ServiceServer service_;

  std::vector<std::shared_ptr<movidius_ncs_lib::NCS>> ncs_handle_;
  ros::NodeHandle nh_;

  int device_index_;
  int log_level_;
  std::string cnn_type_;
  std::string graph_file_path_;
  std::string category_file_path_;
  int network_dimension_;
  std::vector<float> mean_;
  float scale_;
  int top_n_;
};
}  // namespace movidius_ncs_image

#endif  // MOVIDIUS_NCS_IMAGE_NCS_SERVER_H
