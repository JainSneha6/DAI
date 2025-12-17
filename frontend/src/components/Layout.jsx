import React from "react";
import Sidebar from "./Sidebar";

export default function Layout({ children, showSidebar = true }) {
    if (!showSidebar) {
        return <>{children}</>;
    }

    return (
        <div className="flex min-h-screen bg-gradient-to-br from-[#1b1f30] via-[#1a2238] to-[#111827]">
            <Sidebar />
            <main className="flex-1">
                {children}
            </main>
        </div>
    );
}