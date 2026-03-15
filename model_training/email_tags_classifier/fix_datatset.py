import pandas as pd
import random

df = pd.read_csv(r"dataset/email_intent_datasets.csv")

df["tags"] = df["tags"].apply(lambda x: [t.strip() for t in x.split(",")])

# tag specific keywords
tag_keywords = {
    "Demo":["schedule a demo","product demo","live demo","demo session"],
    "Sales":["buy","purchase","sales inquiry","interested in buying"],
    "Support":["need help","support needed","assist me","customer support"],
    "Technical":["API integration","technical issue","system integration","developer setup"],
    "Complaint":["not working","very disappointed","facing issue","system crashed"],
    "Pricing":["pricing details","cost information","enterprise pricing","price quote"],
    "Unsubscribe":["unsubscribe","remove me from mailing list","stop emails"],
    "Question":["i have a question","need clarification","want to ask"],
    "General":["more information","tell me about your service","general inquiry"]
}

def augment_email(email, tags):

    new_email = email

    for tag in tags:
        if tag in tag_keywords:
            if random.random() < 0.7:
                new_email = new_email + " " + random.choice(tag_keywords[tag])

    return new_email


augmented_rows = []

for _, row in df.iterrows():

    email = row["email_text"]
    tags = row["tags"]

    # keep original
    augmented_rows.append({
        "email_text": email,
        "tags": ",".join(tags)
    })

    # generate variations
    for _ in range(2):

        new_email = augment_email(email, tags)

        augmented_rows.append({
            "email_text": new_email,
            "tags": ",".join(tags)
        })


new_df = pd.DataFrame(augmented_rows)

print("Old dataset:",len(df))
print("New dataset:",len(new_df))

new_df.to_csv(r"dataset/email_intent_dataset_fixed.csv",index=False)

print("New dataset saved.")