import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Home from "../components/Home";
import Formpage from "../components/Formpage";
import Results from "../components/Results";
import Profile from "../components/Profile";
import CartPage from "../components/CartPage";

const AUTH_TOKEN_KEY = "authToken";

function ProtectedRoute({ children }) {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    if (!token) {
        return <Navigate to="/" replace />;
    }
    return children;
}

export default function Router() {
    return (
        <div>
            <BrowserRouter>
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route
                        path="/form"
                        element={
                            <ProtectedRoute>
                                <Formpage />
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/results"
                        element={
                            <ProtectedRoute>
                                <Results />
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/profile"
                        element={
                            <ProtectedRoute>
                                <Profile />
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/cart"
                        element={
                            <ProtectedRoute>
                                <CartPage />
                            </ProtectedRoute>
                        }
                    />
                    
                </Routes>
            </BrowserRouter>
        </div>
    );
}
