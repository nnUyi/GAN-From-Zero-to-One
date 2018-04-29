# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/27/2018
github: https://github.com/nnUyi

Requirements:
    python3.*
'''

# class definition
'''
    syntax:
        class class_name:
            class suites
            
'''

'''
-----------------------------------------------------------------
                       class definition
-----------------------------------------------------------------
'''
class Student:
    'Student Class'
    # constructor
    def __init__(self, name, age, major):
        self.name = name
        self.age = age
        self.major = major
    # destructor
    def __del__(self):
        print('delete Student')
    
    def __repr__(self):
        return 'Student repr'
    
    def __str__(self):
        return 'Student str'
    
    def print_repr(self):
        print(self.__repr__())
    
    def print_str(self):
        print(self.__str__())
    
    def get_name_attr(self):
        return self.name

    def get_age_attr(self):
        return self.age

    def get_major_attr(self):
        return self.major

    def set_name_attr(self, name):
        self.name = name
        
    def set_age_attr(self, age):
        self.age = age
        
    def set_major_attr(self, major):
        self.major = major



'''
-----------------------------------------------------------------
                       class instance
-----------------------------------------------------------------
'''
print('class instance'+'-'*20)
stu = Student('yyz', 24, 'cs')
# access class property
stu_name = stu.name
stu_age = stu.age
stu_major = stu.major
# access class method
stu_name_ = stu.get_name_attr()
stu_age_ = stu.get_age_attr()
stu_major_ = stu.get_major_attr()
# print stu info
print('stu info: name.{}, age.{}, major.{}'.format(stu_name, stu_age, stu_major))
print('stu_ info: name.{}, age.{}, major.{}'.format(stu_name_, stu_age_, stu_major_))

print('\n')



'''
-----------------------------------------------------------------
               class build-in attributes and methods
-----------------------------------------------------------------
'''
# properties
'''
    __dict__: properties of class
    __doc__ : doc string of class
    __module__: class module
'''
print('class build-in properties and methods'+'-'*20)
student = Student('yyz', 24, 'cs')
print('__doc__:{}'.format(student.__doc__))
print('__module__:{}'.format(student.__module__))
print('__dict__:{}'.format(student.__dict__))

# methods
'''
    __init__(self)  constructor
    __del__(self)   destructor  
    __repr__(self)  avariable printing string of obj 
    __str__(self)   avariable string of obj
    __cmp__(self)   object comparision
'''
student.print_str()
student.print_repr()

print('\n')



'''
-----------------------------------------------------------------
                       class inherit
-----------------------------------------------------------------
'''
'''
    syntax:
        class class_name(parent1[,parent2, ...]):
            class suites
            
'''
print('class inherit'+'-'*20)
class parent:
    def __init__(self):
        print('parent instance')
    
    def __del__(self):
        print('delete parent')
    
    def myMethod(self):
        print('in parent class')

class child(parent):
    def __init__(self):
        # initialize parent class
        super(child, self).__init__()
        print('child instance')

    def __del__(self):
        print('delete child')
        
    # method inherit        
    def myMethod(self):
        print('in child class')

# instance
parent_instance = parent()
child_instance = child()
parent_instance.myMethod()
child_instance.myMethod()

print('\n')



'''
-----------------------------------------------------------------
                 personal private & public
-----------------------------------------------------------------
'''
'''
    official:   __name__
    private:    __name
    personal:   _name
    public:     name   
'''
print('personal private & public'+'-'*20)
class Teacher:
    'Teacher Class'
    __name = None
    def __init__(self):
        print('teacher instance')
    
    def public_method(self):
        print('This is public method')
        
    def _personal_method(self):
        print('This is personal method')
        
    def __private_method(self):
        print('This is private method')

# instance
teacher = Teacher()
teacher.public_method()
teacher._personal_method()
# private method
try:
    teacher.__private_method()
except AttributeError:
    print('no such attr in teacher class')
# private attribute
try:
    print('teacher name:{}'.format(teacher.__name))
except AttributeError:
    print('no such attr in teacher class')
