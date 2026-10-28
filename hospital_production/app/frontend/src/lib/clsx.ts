/** Tiny helper — avoids pulling in the full clsx package */
export function clsx(...args: (string | undefined | false | null)[]): string {
  return args.filter(Boolean).join(" ");
}

