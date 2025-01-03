# Age Group categorization

age = int(input("Enter your age: "))

if(age<12):
    print("You are still a child.")
elif(12<age<18):
    print("You are a teenager.")
elif(18<=age<60):
    print("You are adult.")
else:
    print("You are older than most of the public, go on the path of spiritualit.")
