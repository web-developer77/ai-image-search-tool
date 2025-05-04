"use client";

import { useState } from "react";
import { Search } from "lucide-react";
import axios from "axios";

const API_URL = process.env.BACKEND_API_URL || "http://localhost:8000";

export default function ImageSearchPage() {
  const [images, setImages] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    const formData = new FormData(e.currentTarget);
    const prompt = formData.get("prompt") as string;

    try {
      const response = await axios.post(`${API_URL}/api/search`, {
        prompt: prompt,
      });
      setImages(response.data);
    } catch (error) {
      console.error("Error searching images:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center bg-gray-50 p-4">
      <div className="w-full max-w-md space-y-6 mt-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">
            Image Search
          </h1>
          <p className="mt-2 text-sm text-gray-600">
            Enter a prompt to search for related images
          </p>
        </div>

        <form onSubmit={handleSubmit} className="mt-8 space-y-4">
          <div className="rounded-md shadow-sm">
            <div>
              <label htmlFor="prompt" className="sr-only">
                Search prompt
              </label>
              <textarea
                id="prompt"
                name="prompt"
                required
                className="relative block w-full rounded-md border-0 py-3 px-4 text-gray-900 ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm"
              />
            </div>
          </div>

          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative flex w-full justify-center rounded-md bg-indigo-600 py-3 px-4 text-sm font-semibold text-white hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:opacity-50"
            >
              <span className="absolute inset-y-0 left-0 flex items-center pl-3">
                <Search
                  className="h-5 w-5 text-indigo-500 group-hover:text-indigo-400"
                  aria-hidden="true"
                />
              </span>
              {loading ? "Searching..." : "Search Images"}
            </button>
          </div>
        </form>

        <div className="text-center text-sm text-gray-500">
          <p>Search for any type of image using descriptive prompts</p>
        </div>
      </div>

      {/* Image Results Section */}
      <div className="w-full max-w-6xl mt-12 mb-16">
        <h2 className="text-2xl font-semibold text-gray-800 mb-6">
          Search Results
        </h2>

        {/* Image Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mt-8">
          {images.length > 0 ? (
            images.map((url, index) => (
              <div
                key={index}
                className="bg-white rounded-lg overflow-hidden shadow-sm border border-gray-200 hover:shadow-md transition-shadow"
              >
                <div className="aspect-square bg-gray-100 flex items-center justify-center">
                  <img
                    src={url}
                    alt={`Search result ${index + 1}`}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="p-3">
                  <h3 className="text-sm font-medium text-gray-800 truncate">
                    Result {index + 1}
                  </h3>
                </div>
              </div>
            ))
          ) : (
            <div className="col-span-full text-center text-gray-500">
              {loading
                ? "Searching for images..."
                : "No images found. Try searching for something!"}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
