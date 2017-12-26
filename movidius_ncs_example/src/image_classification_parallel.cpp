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

#include <vector>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>

#include <ros/ros.h>
#include <movidius_ncs_msgs/ClassifyObject.h>

std::vector<std::string> getImagePath(std::string image_dir)
{
  std::vector<std::string> files;

  DIR *dir;
  struct dirent *ptr;

  if ((dir=opendir(image_dir.c_str())) == NULL)
  {
    perror("Open Dir error...");
    exit(1);
  }

  while ((ptr=readdir(dir)) != NULL)
  {
    if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)
      continue;
    else if(ptr->d_type == 8)
      files.push_back(image_dir + ptr->d_name);
  }
  closedir(dir);

  return files;
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "movidius_ncs_example");

  /*
  if (argc != 2)
  {
    ROS_INFO("Usage: rosrun movidius_ncs_example movidius_ncs_example_image_classification <image_path>");
    return -1;
  }
  */
  
  char basePath[100] = "/opt/movidius/ncappzoo/data/images/";
  //char basePath[100] = "/home/chao/test/";
  std::vector<std::string> images_path=getImagePath(basePath);
  for (unsigned int i=0; i < images_path.size(); i++)
  {
      ROS_INFO("image: %s",images_path[i].c_str());
  }

  ros::NodeHandle n;
  ros::ServiceClient client;
  client = n.serviceClient<movidius_ncs_msgs::ClassifyObject>("/movidius_ncs_image/classify_object");
  movidius_ncs_msgs::ClassifyObject srv;
  srv.request.image_path = images_path;
  
  //
  for (unsigned int i = 0; i < srv.request.image_path.size(); i++)
  {
    ROS_INFO("%d: %s", i, srv.request.image_path[i].c_str());
  }
  ROS_INFO("ready to call callback");
  
  //
  struct timeval inference_start, inference_end;
  gettimeofday(&inference_start, NULL);
  
  if (!client.call(srv))
  {
    ROS_ERROR("failed to call service ClassifyObject");
    return 1;
  }
  

  //
  ROS_INFO("done for call back");
  //
  
  float total_inference_time_ms = 0;
  
  ROS_INFO("example: size of objects is %lu", srv.response.objects.size());

  for (unsigned int i = 0; i < srv.response.objects.size(); i++)
  {
    for (unsigned int j = 0; j < srv.response.objects[i].objects_vector.size(); j++)
    {
      ROS_INFO("%d: object: %s\nprobability: %lf%%", j,
               srv.response.objects[i].objects_vector[j].object_name.c_str(),
               srv.response.objects[i].objects_vector[j].probability * 100);
    }
    ROS_INFO("inference time: %fms", srv.response.objects[i].inference_time_ms);
    total_inference_time_ms = total_inference_time_ms + srv.response.objects[i].inference_time_ms;
  }
  gettimeofday(&inference_end, NULL);
  float time_used = (inference_end.tv_sec - inference_start.tv_sec) * 1000000 + (inference_end.tv_usec - inference_start.tv_usec);
  ROS_INFO("total inference time of %lu pics is %fms", images_path.size(), time_used / 1000);

  return 0;
}
