def pprint_args(args_namespace):
    pp = "\n".join([f"\t-- {key}: {getattr(args_namespace, key)}" for key in vars(args_namespace)])
    print(f"==========================================================\n"
          f"ARGUMENTS:\n-----------\n"
          f"{pp}\n"
          f"==========================================================\n")