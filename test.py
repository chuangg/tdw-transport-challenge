import pickle


def convert_action(action_):
    if action_["type"] == 1:

    return []


def main():
    with open("/Users/Abhi.B@ibm.com/rotation_projects/extra_projects/int_phys_human_study/to_send/tdw_crash/submissions_3/actions_history.pickle", "rb") as fp:
        data = pickle.load(fp)
        print(len(data["episode_no_3"]["actions"]))
    output = {
        "seed": data["episode_no_3"]["seed"]["seed"],
        "actions": data["episode_no_3"]["actions"],
    }
    output["episode_no_3"]["actions"] = [convert_action(e) for e in data["episode_no_3"]["actions"]]


if __name__ == '__main__':
    main()


