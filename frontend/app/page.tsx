import Link from "next/link";
import SearchBar from "@/components/SearchBar";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-white px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center py-20">
          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
            Aion Analytics
          </h1>
          <p className="text-xl text-gray-400 mb-12">
            AI-powered stock analysis and predictions
          </p>
          
          <SearchBar />
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16">
            <Link
              href="/bots"
              className="p-6 rounded-2xl border border-gray-800 bg-gray-900/70 hover:bg-gray-800/70 transition-colors"
            >
              <h2 className="text-xl font-semibold mb-2">Bots</h2>
              <p className="text-gray-400">Manage trading bots and strategies</p>
            </Link>
            
            <Link
              href="/insights"
              className="p-6 rounded-2xl border border-gray-800 bg-gray-900/70 hover:bg-gray-800/70 transition-colors"
            >
              <h2 className="text-xl font-semibold mb-2">Insights</h2>
              <p className="text-gray-400">View AI-powered market insights</p>
            </Link>
            
            <Link
              href="/profile"
              className="p-6 rounded-2xl border border-gray-800 bg-gray-900/70 hover:bg-gray-800/70 transition-colors"
            >
              <h2 className="text-xl font-semibold mb-2">Profile</h2>
              <p className="text-gray-400">Track your portfolio and investments</p>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
