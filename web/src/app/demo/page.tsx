import { redirect } from "next/navigation";

/** Legacy /demo → production /chat */
export default function DemoRedirectPage() {
  redirect("/chat");
}
