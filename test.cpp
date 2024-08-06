#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <sycl/sycl.hpp>


int main() {
  
  size_t num_items = 100000;
  std::vector<int> vec_list(num_items);


  auto gpu = sycl::device();//sycl::device(sycl::gpu_selector_v);
  std::cout << gpu.get_info<sycl::info::device::name>() << std::endl;

  using Vector3 = sycl::vec<double, 3>;
  using EigenVector3 = Eigen::Matrix<double, 3, 1>;
  //Vector3 x = {1.0, 2.0, 3.0};



  constexpr size_t N = 20;
  std::vector<Vector3> stacked{};
  for(int i=0; i < N; i++)
  {
    stacked.push_back(Vector3(i, i+1, i+2));
  }

  sycl::queue q{gpu};
  sycl::buffer stackedB(stacked);

  q.submit(
    [&](sycl::handler &h)
    {
      sycl::accessor stacked_accessor(stackedB, h);
      h.parallel_for(N, [=](sycl::id<1> i)
        {
          auto sycl_vec = stacked_accessor[i];
          EigenVector3 v(sycl_vec.x(), sycl_vec.y(), sycl_vec.z());
          v += EigenVector3(100, 100, 100);
          stacked_accessor[i] = Vector3(v(0,0), v(1,0), v(2,0));
        }
      );
    }
  );

  q.wait();
  std::cout << "UPDATING VIA GPU" << std::endl;

  sycl::host_accessor host(stackedB);
  for(int i=0; i < N; i++)
  {
    Vector3 x = host[i];
    std::cout << "x: " << std::endl << x.x() << " " << x.y() << " " << x.z() << std::endl;
  }

}

