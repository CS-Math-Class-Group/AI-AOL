import { createContext } from "react";
import { type UserInfo } from "../model/user";

export const userContext = createContext<UserInfo | null>(null);