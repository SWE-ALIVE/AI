def turn_data_to_str(data):
    context = []
    for slot_value in data["slot_values"]:
        context.append(
            f"{slot_value['domain']}-{slot_value['slot']}: {slot_value['value']}"
        )

    context_str = "[context] " + ", ".join(context)
    system_str = f"[system] {data['dialog']['sys'][0]}"
    user_str = f"Q: [user] {data['dialog']['usr'][0]}"

    return "\n".join([context_str, system_str, user_str])
