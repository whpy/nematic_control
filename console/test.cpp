#include <iostream>

using namespace std;

void func()
{
	cout << "outside" << endl;
}

class t
{
	public:
		t()
		{}
		void func()
		{
			cout << "inside" << endl;
		}

		void func1()
		{
			cout << "func1 run" << endl;
			func();
			::func();
		}
};

int main()
{
	t test = t();
	test.func1();
	func();
	return 0;
}
