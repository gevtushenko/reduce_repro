#include <iostream>

extern float Reduce1024x100();
int main() {
  auto ret = Reduce1024x100();
  std::cout << ret << std::endl;
  return 0;
}
