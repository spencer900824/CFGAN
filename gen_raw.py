import random

def gen_user():
    id = ''.join(random.choice('0123456789') for i in range(10))
    gender = random.choice(['M', 'F', 'N'])
    age = random.randint(20, 80)
    return [id, gender, age]

def gen_merchant():
    id = random.randint(10000, 20000)
    category = random.randint(1, 5)
    return [id, category, 0]


uN = 10000
mN = 10000

users = [ gen_user() for i in range(uN) ]
merchants = [ gen_merchant() for i in range(mN) ]

raw = 10000
with open('raw.data', 'a') as f:
    for i in range(raw):
        user = random.choice(users)
        merchant = random.choice(merchants)
        data = user.extend(merchant)
        f.write('      '.join(data))+'\n')
