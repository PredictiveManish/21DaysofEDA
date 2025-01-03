# Reverse a string
new = ""
string = str(input("Enter any string to reverse: "))
for char in string:
    new = char + new
print(new)