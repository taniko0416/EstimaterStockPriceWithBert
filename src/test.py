import datetime
class Test(object):
    def __getitem__(self, x):
        the_hour = datetime.datetime.now().hour
        if x == "greeting":
            if 0 <= the_hour < 6:
                return "Don't to speak to me"
            elif 6 <= the_hour < 10:
                return "Goodmorning"
            elif 10 <= the_hour < 13:
                return "Goodnoon"
            elif 13 <= the_hour < 17:
                return "Goodafternoon"
            else:
                return "Goodevening"
        elif x == "sit":
            return "down"
        elif x == "stand":
            return "up"
        else:
            return "What?"
test = Test()
print("greeting: "+test["greeting"])
print("sit: "+test["sit"])
print("stand: "+test["stand"])
print(" : "+test[""])