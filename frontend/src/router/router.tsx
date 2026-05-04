import { createBrowserRouter } from "react-router-dom";
import HomePage from "../pages/HomePage";
import CollectionPage from "../pages/CollectionPage";
import MusicPage from "../pages/MusicPage";
import ConvertPage from "../pages/ConvertPage";
import LoginPage from "../pages/LoginPage";
import RegisterPage from "../pages/RegisterPage";

export const router = createBrowserRouter([
    {
        path:"/",
        element: <HomePage/>
    },
    {
        path:"/home",
        element: <HomePage/>
    },
    {
        path:"/collection",
        element: <CollectionPage/>,
        children: [
            {
                path:"/:musicid",
                element: <MusicPage/>
            }
        ]
    },
    {
        path:"/convert",
        element: <ConvertPage/>
    },
    {
        path: "/login",
        element: <LoginPage/>
    },
    {
        path: "/register",
        element: <RegisterPage/>
    },
])